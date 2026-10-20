"""Run with: streamlit run app.py"""
from dataclasses import asdict, replace
import json
import os
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import streamlit as st
from data_access import load
from model import Inputs, calculate

st.set_page_config(page_title='Foreign cost of capital',page_icon='🌐',layout='wide')
st.markdown('''<style>.stApp{background:#fff}h1,h2,h3{color:#a40000!important}
[data-testid="stMetricValue"]{color:#17365d}div[data-testid="stSidebar"]{background:#f5f7fa}</style>''',unsafe_allow_html=True)
st.title('Foreign cost of capital')
st.markdown('Compare currency conversion with a direct country-risk-adjusted estimate of WACC.')

@st.cache_data(ttl=86400,show_spinner=False)
def datasets(live):
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(load,k,live) for k in ('industries','countries')]
        return [f.result() for f in futures]

with st.sidebar:
    st.header('Business and financing')
    live=st.checkbox('Download latest Damodaran tables',value=os.getenv('WACC_OFFLINE')!='1')
    if st.button('Refresh data'):
        datasets.clear()
    st.caption('If a download fails, the app uses its dated bundled snapshot.')
try:
    with st.spinner('Loading industry and country data…'):
        (industries,imeta),(countries,cmeta)=datasets(live)
except Exception as exc:
    st.error(f'Data could not be loaded: {exc}');st.stop()
for meta in (imeta,cmeta):
    if meta['error']:
        st.warning('A Damodaran download failed. A bundled snapshot is being used; see Data sources for its date.')

with st.sidebar:
    names=industries.industry.tolist()
    industry=st.selectbox('Industry (US comparables)',names,index=names.index('Food Processing') if 'Food Processing' in names else 0)
    cnames=countries.country.tolist()
    country=st.selectbox('Country of operations',cnames,index=cnames.index('Portugal') if 'Portugal' in cnames else 0)
    leverage=st.number_input('Target debt/equity (%)',0.0,1000.0,100.0,10.0)/100
    st.caption('100% D/E means 50% debt and 50% equity. Use market-value weights.')
    denomination=st.selectbox('Valuation currency',['US dollars','Foreign currency'])
    currency='USD' if denomination=='US dollars' else st.text_input('Foreign currency code','EUR').strip().upper()
    method=st.selectbox('Currency conversion method',['Risk-free interest rates','Expected inflation']) if denomination!='US dollars' else 'Risk-free interest rates'

ir=industries.loc[industries.industry==industry].iloc[0]
cr=countries.loc[countries.country==country].iloc[0]
us=countries.loc[countries.country.str.strip().isin(['United States','United States of America'])]
if us.empty: st.error('United States row is missing from the country dataset.');st.stop()
us=us.iloc[0]
base=float((countries.erp-countries.crp).median())
if ((countries.erp-countries.crp)-base).abs().max()>.0003:
    st.warning('Country rows imply inconsistent mature-market premiums. Review the data and premium assumption below.')
if 'bank' in industry.lower() or 'insurance' in industry.lower():
    st.info('For financial firms, debt is part of operations. Interpret a conventional enterprise WACC with care.')

with st.expander('Rates, premiums, and borrowing assumptions',expanded=True):
    st.caption('Risk-free rates and company spreads below are editable illustrative assumptions, not live market quotes. Country premiums and tax defaults come from Damodaran.')
    a,b,c=st.columns(3)
    rfusd=a.number_input('USD default-free rate (%)',-20.0,100.0,4.0,.1)/100
    if denomination=='US dollars':
        rff=rfusd; infus=inflocal=.02
    elif method=='Risk-free interest rates':
        rff=b.number_input('Foreign-currency default-free rate (%)',-20.0,100.0,4.0,.1)/100
        infus=inflocal=.02
        b.caption('Use a default-free rate, not an unadjusted risky sovereign yield.')
    else:
        infus=b.number_input('Expected US inflation (%)',-20.0,1000.0,2.0,.1)/100
        inflocal=c.number_input('Expected foreign inflation (%)',-20.0,1000.0,2.0,.1)/100
        rff=(1+rfusd)*(1+inflocal)/(1+infus)-1
        st.caption(f'Foreign default-free rate implied by common real rates: {rff:.2%}.')
    erpus=a.number_input('US total equity premium (%)',0.0,100.0,float(us.erp*100),.1,key='us_erp_'+str(us.erp))/100
    mature=b.number_input('Mature-market equity premium (%)',0.0,100.0,round(base*100,6),.1,key='mature_'+str(base))/100
    cp=c.number_input('Country equity-risk premium (%)',0.0,100.0,float(cr.crp*100),.1,key='crp_'+country)/100
    tax_default=float(cr.tax) if pd.notna(cr.tax) and 0<=cr.tax<=1 else .25
    if pd.isna(cr.tax): st.warning('No usable country tax rate: the 25% assumption requires review.')
    tax=a.number_input('Marginal tax rate (%)',0.0,100.0,tax_default*100,1.0,key='tax_'+country)/100
    spread=b.number_input('Company borrowing spread (%)',0.0,100.0,2.0,.1)/100
    b.caption('Total spread above the selected default-free rate; include sovereign exposure here if applicable. No sovereign spread is added automatically.')
    extended=c.checkbox('Include country risk before converting Approach 1',value=False)
    c.caption('Off: US benchmark, as in the slides. On: use mature-market premium + country premium in the USD starting return.')
    beta=a.number_input('Industry equity beta',-5.0,10.0,float(ir.beta),.05,key='beta_'+industry)
    ide=b.number_input('Industry observed D/E (%)',0.0,10000.0,float(ir.de*100),1.0,key='ide_'+industry)/100
    st.caption(f'Industry sample: {int(ir.firms)} firms. Beta is unlevered and relevered without a tax adjustment, following your slides.')

p=Inputs(beta,ide,leverage,tax,rfusd,rff,erpus,mature,cp,spread,infus,inflocal,
         'inflation' if method=='Expected inflation' else 'rates',extended)
try: r=calculate(p)
except ValueError as exc: st.error(str(exc));st.stop()
st.caption(f'{industry} · {country} · Nominal {currency} cash flows and discount rates · Target D/E {leverage:.0%}')
left,right=st.columns(2)
with left:
    st.subheader('1 · Convert a USD return')
    st.metric('WACC',f'{r["wacc1"]:.2%}')
    st.metric('Cost of equity',f'{r["equity1"]:.2%}')
with right:
    st.subheader('2 · Direct market estimate')
    st.metric('WACC',f'{r["wacc2"]:.2%}')
    st.metric('Cost of equity',f'{r["equity2"]:.2%}')
st.info('Approach 1 converts a US benchmark into the valuation currency. Approach 2 explicitly prices operating-country equity risk. Their difference is not a pure measure of currency risk.')

explain, sensitivity, sources = st.tabs(['Calculation steps','Leverage sensitivity','Data sources'])
with explain:
    st.markdown('**1. Remove industry leverage and apply target leverage**')
    st.latex(r'\beta_U=\frac{\beta_{E,industry}}{1+L_{industry}},\qquad\beta_E=\beta_U(1+L)')
    st.write(f'Unlevered beta = {beta:.4f} / (1 + {ide:.4f}) = {r["beta_unlevered"]:.4f}; target equity beta = {r["beta_equity"]:.4f}.')
    st.markdown('**2. Convert the USD cost of equity**')
    st.latex(r'k_{E,\$}=r_{f,\$}+\beta_E ERP_{reference}')
    if method=='Expected inflation':
        st.latex(r'k_{E,F}^{(1)}=(1+k_{E,\$})\frac{1+\pi_F^e}{1+\pi_\$^e}-1')
    else:
        st.latex(r'k_{E,F}^{(1)}=(1+k_{E,\$})\frac{1+r_{f,F}}{1+r_{f,\$}}-1')
    st.write(f'USD equity return: {r["equity_usd"]:.4%}; conversion factor: {r["conversion_factor"]:.6f}; foreign equity return: {r["equity1"]:.4%}.')
    st.markdown('**3. Estimate equity return directly**')
    st.latex(r'k_{E,F}^{(2)}=r_{f,F}+\beta_E(ERP_M+CRP_c)')
    st.write(f'{rff:.4%} + {r["beta_equity"]:.4f} × ({mature:.4%} + {cp:.4%}) = {r["equity2"]:.4%}.')
    st.caption('Damodaran estimates CRP by multiplying the sovereign default spread by relative equity/bond volatility. The app uses his published CRP, avoiding an additional scaling step.')
    st.markdown('**4. Calculate borrowing costs and financing weights**')
    st.latex(r'k_D=r_{f,F}+s_{company},\quad k_{D,aftertax}=k_D(1-T),\quad w_E=\frac1{1+L},\quad w_D=\frac L{1+L}')
    st.write(f'Pretax debt: {r["debt_pretax"]:.4%}; after-tax debt: {r["debt_aftertax"]:.4%}; equity weight: {r["weight_equity"]:.2%}; debt weight: {r["weight_debt"]:.2%}.')
    st.latex(r'WACC_F^{(j)}=w_E k_{E,F}^{(j)}+w_D k_D(1-T)')
    for j in (1,2):
        st.write(f'Approach {j}: {r["weight_equity"]:.4f} × {r[f"equity{j}"]:.4%} + {r["weight_debt"]:.4f} × {r["debt_aftertax"]:.4%} = {r[f"wacc{j}"]:.4%}.')
    st.caption('Assumes debt beta is zero, interest deductions are usable, and the target capital structure applies to the cash flows being valued. No additional exchange-rate risk premium is imposed, consistent with the slides.')
with sensitivity:
    st.write('Borrowing spread held constant. These curves do not identify an optimal capital structure.')
    end=max(3.0,leverage)
    rows=[]
    for i in range(101):
        lev=end*i/100
        q=calculate(replace(p,target_de=lev))
        rows.append({'D/E (%)':lev*100,'Approach 1 WACC (%)':q['wacc1']*100,'Approach 2 WACC (%)':q['wacc2']*100})
    chart=pd.DataFrame(rows).set_index('D/E (%)')
    st.line_chart(chart,x_label='Debt/equity (%)',y_label='WACC (%)',color=['#a40000','#17365d'])
    st.download_button('Download sensitivity CSV',chart.to_csv().encode(),'wacc_sensitivity.csv','text/csv')
with sources:
    for title,meta in [('Industry betas',imeta),('Country risk premiums',cmeta)]:
        st.markdown(f'**[{title}]({meta["url"]})**')
        st.write(f'{meta["mode"]} · Source date: {meta["source_date"]} · Retrieved (UTC): {meta["retrieved"]}')
        if meta['error']: st.caption(meta['error'])
    st.write('Country default spread:',f'{cr.sovereign_spread:.2%}')
    st.caption('The default spread is shown for context, not automatically added to borrowing costs. Editable rates and spreads are assumptions. Bundled data remain usable offline; refresh downloads when internet access is available.')
    st.dataframe(industries,hide_index=True)
    st.dataframe(countries,hide_index=True)

export={'industry':industry,'country':country,'currency':currency,'inputs':asdict(p),'results':r,'sources':{'industry':imeta,'country':cmeta}}
st.download_button('Download full calculation (JSON)',json.dumps(export,indent=2,allow_nan=False).encode(),'wacc_calculation.json','application/json')
st.caption('Based on Sergio Rebelo’s CostofCapital.pdf. Data: Aswath Damodaran, NYU Stern. Rates are annual; all calculations retain full precision.')
