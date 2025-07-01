import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, r2_score
import streamlit as st
import plotly.express as px
st.set_page_config(layout="wide")

#mudar o path para o necessário
path = r"C:\Users\55859\Desktop\7_semestre\MS905\projetofinal\archive_projfinal\Uncleaned_DS_jobs.csv"
df = pd.read_csv(path)

# rodar o comando abaixo no terminal, corrigindo o caminho
#streamlit run C:\Users\55859\PycharmProjects\PythonProject\projetofinal.py

#st.dataframe(df)

#st.write(len(df['Job Title'].unique()))
#st.write(df['Job Title'].unique())
#st.write(df['Type of ownership'].unique())
#st.write(df['Industry'].unique())

#st.write(len(df['Competitors'].unique()))


db = df.copy()
db['Type of ownership']=db['Type of ownership'].replace(['-1', 'Unknown'], 'Sem informação')

db['Industry'] = db['Industry'].replace(['-1', 'Unknown'], 'Sem informação')
db['Sector'] = db['Sector'].replace(['-1', 'Unknown'], 'Sem informação')
db['Revenue'] = db['Revenue'].replace(['-1', 'Unknown / Non-Applicable'], 'Sem informação')

#db['Number of competitors'] = db['Competitors'].copy()
db['Número de Competidores Listados'] = db['Competitors'].apply(lambda n: len(n.split(', ')) if n  != '-1' else 0)
#st.write(db[['Industry', 'Sector']])
#st.write(db['Number of competitors'].unique())

db['Estado'] = db['Location'].str.split(',').str[-1]
db['Estado'] = db['Estado'].replace('United States', 'Any US State')
db['Estado'] = db['Estado'].replace('Utah', 'UT')
db['Estado'] = db['Estado'].replace('New Jersey', 'NJ')
db['Estado'] = db['Estado'].replace('Texas', 'TX')
db['Estado'] = db['Estado'].replace('California', 'CA')

db['Headquarters'] = db['Headquarters'].replace('New York, 061', 'New York, NY')
db['Estado da Sede'] = db['Headquarters'].str.split(',').str[-1]
db['Estado da Sede'] = db['Estado da Sede'].replace('United States', 'Any US State')
db['Estado da Sede'] = db['Estado da Sede'].replace('Utah', 'UT')
db['Estado da Sede'] = db['Estado da Sede'].replace('New Jersey', 'NJ')
db['Estado da Sede'] = db['Estado da Sede'].replace('Texas', 'TX')
db['Estado da Sede'] = db['Estado da Sede'].replace('California', 'CA')
db['Estado da Sede'] = db['Estado da Sede'].replace('-1', 'Sem informação')
db['Estado da Sede'] = db['Estado da Sede'].replace('061', 'NY')

db['Estimativa de Salário'] = db['Salary Estimate'].str.split('(').str[0]
#db['Estimativa de Salário'] = db['Salary Estimate'].str.split().str[0]
db['Estimativa de Salário'] = db['Estimativa de Salário'].str.replace('$', '')
db['Estimativa de Salário'] = db['Estimativa de Salário'].str.replace('K', '')
db['Salário Mínimo'] = db['Estimativa de Salário'].str.split('-').str[0].astype(float)
db['Salário Máximo'] = db['Estimativa de Salário'].str.split('-').str[-1].astype(float)
db['Salário Médio (U$ K)'] = (db['Salário Mínimo']+db['Salário Máximo'])/2
db['Número de Funcionários'] = db['Size'].str.replace('employees', '')
db['Número de Funcionários'] = db['Número de Funcionários'].str.replace('to', 'a')
db['Número de Funcionários']=db['Número de Funcionários'].replace(['-1', 'Unknown'], 'Sem informação')

#st.write(len(db['Size'].unique()))
#st.write(db['Salário Médio (U$ K)'].unique())

db['Data Science'] = db['Job Title'].str.contains('Data Science').replace(True, 1).replace(False,0)
db['Mesmo estado da Sede'] = (db['Estado']==db['Estado da Sede']).replace(True, 1).replace(False,0)
db['Cientista de Dados 0']=db['Job Title'].str.contains('Data Scientist').replace(True, 1).replace(False,0)
db['Cientista de Dados 1']=db['Job Title'].str.contains('Data Science').replace(True, 1).replace(False,0)
db['Cientista de ML']=db['Job Title'].str.contains('Machine Learning Scientist').replace(True, 1).replace(False,0)
db['Engenheiro ML']=db['Job Title'].str.contains('Machine Learning Engineer').replace(True, 1).replace(False,0)
db['Analista de Dados']=db['Job Title'].str.contains('Analyst').replace(True, 1).replace(False,0)
db['Engenheiro de Dados']=db['Job Title'].str.contains('Data Engineer').replace(True, 1).replace(False,0)
db['Gerencial0']=db['Job Title'].str.contains('Manager').replace(True, 1).replace(False,0)
db['Gerencial1']=db['Job Title'].str.contains('Director').replace(True, 1).replace(False,0)
db['Cargo Senior 0']=db['Job Title'].str.contains('Senior').replace(True, 1).replace(False,0)
db['Cargo Sr']=db['Job Title'].str.contains('Sr').replace(True, 1).replace(False,0)
db['Cargo senior']=db['Job Title'].str.contains('senior').replace(True, 1).replace(False,0)
db['Cargo sr']=db['Job Title'].str.contains('sr').replace(True, 1).replace(False,0)
db['ML']=db['Job Title'].str.contains('Machine Learning').replace(True, 1).replace(False,0)
db['Data Modeler']=db['Job Title'].str.contains('Data Modeler').replace(True, 1).replace(False,0)
db['AI']=db['Job Title'].str.contains('AI').replace(True, 1).replace(False,0)
db['Negócios']=db['Job Title'].str.contains('Business').replace(True, 1).replace(False,0)
db['Cientista0']=db['Job Title'].str.contains('Scientist').replace(True, 1).replace(False,0)
db['Cientista1']=db['Job Title'].str.contains('SCIENTIST').replace(True, 1).replace(False,0)
db['Engenheiro']=db['Job Title'].str.contains('Engineer').replace(True, 1).replace(False,0)
db['Cargo VP0']=db['Job Title'].str.contains('VP').replace(True, 1).replace(False,0)
db['Cargo VP1']=db['Job Title'].str.contains('Vice President').replace(True, 1).replace(False,0)
db['Cientista da Computação']=db['Job Title'].str.contains('Computer Scientist').replace(True, 1).replace(False,0)
db['Arquiteto de Dados']=db['Job Title'].str.contains('Data Architect').replace(True, 1).replace(False,0)
db['Cientista de Dados 2'] = np.maximum(db['Cientista de Dados 0'], db['Cientista de ML'], db['Cientista de Dados 1'])
db['Cientista de Dados3'] = np.maximum(db['Engenheiro ML'], db['Cientista de Dados 2'])
db['Cientista de Dados4'] = np.maximum(db['Data Modeler'], db['Cientista de Dados3'])
db['Cientista de Dados'] = np.maximum(db['Data Science'], db['Cientista de Dados4'])
db['Cargo Senior 1'] = np.maximum(db['Cargo Senior 0'], db['Cargo Sr'])
db['Cargo VP'] = np.maximum(db['Cargo VP0'], db['Cargo VP1'])
db['Gerencial'] = np.maximum(db['Gerencial1'], db['Gerencial0'])
db['Cientista'] = np.maximum(db['Cientista1'], db['Cientista0'])
db['Cargo senior 1'] = np.maximum(db['Cargo sr'], db['Cargo senior'])
db['Cargo Senior'] = np.maximum(db['Cargo Senior 1'], db['Cargo senior 1'])
db['Revenue'] = db['Revenue'].str.replace('(USD)', '')
db['Revenue'] = db['Revenue'].str.replace('billion', 'Bi')
db['Revenue'] = db['Revenue'].str.replace('million', 'MM')
db['Revenue'] = db['Revenue'].str.replace('Less than', 'Menor que')
db['Revenue'] = db['Revenue'].str.replace('to', 'a')
db['Revenue'] = db['Revenue'].str.replace('$', '')
db['Revenue'] = db['Revenue'].str.replace('Sem informação', '-1')
db['Revenue'] = db['Revenue'].str.replace('100 a 500 MM', '300')
db['Revenue'] = db['Revenue'].str.replace('1 a 2 Bi', '1500')
db['Revenue'] = db['Revenue'].str.replace('10+ Bi', '10000')
db['Revenue'] = db['Revenue'].str.replace('2 a 5 Bi', '3500')
db['Revenue'] = db['Revenue'].str.replace('500 MM a 1 Bi', '725')
db['Revenue'] = db['Revenue'].str.replace('5 a 10 Bi', '7500')
db['Revenue'] = db['Revenue'].str.replace('10 a 25 Bi', '17500')
db['Revenue'] = db['Revenue'].str.replace('10 a 25 MM', '17.5')
db['Revenue'] = db['Revenue'].str.replace('25 a 50 MM', '37.5')
db['Revenue'] = db['Revenue'].str.replace('50 a 100 MM', '75')
db['Revenue'] = db['Revenue'].str.replace('1 a 5 MM', '6')
db['Revenue'] = db['Revenue'].str.replace('5 a 10 MM', '7.5')
db['Revenue'] = db['Revenue'].str.replace('Menor que 1 MM', '1')
#st.dataframe(db)

dbf1 = db.drop(columns = ['Cientista de Dados 0', 'Cientista de ML', 'Salary Estimate', 'Job Description', 'Company Name', 'Location'
                         , 'Headquarters', 'Size', 'Cargo Senior 0', 'Cargo Sr', 'Cargo senior', 'Cargo sr', 'Cargo Senior 1', 'Cargo senior 1',
                         'Competitors','Industry', 'index', 'Job Title'
                          #,'Cargo Senior'
                     ,'Estimativa de Salário', 'Salário Mínimo', 'Salário Máximo'
                      ,'Cientista de Dados 1', 'Engenheiro ML', 'Cientista de Dados 2', 'Cargo VP1', 'Cargo VP0', 'Data Science'
    #, 'Cargo VP', 'ML', 'AI', 'Negócios'
                         ,'Data Modeler', 'Gerencial0', 'Gerencial1', 'Cientista0', 'Cientista1', 'Cientista de Dados3', 'Cientista de Dados4', 'Estado', 'Estado da Sede'])
#st.write(dbf['Salário Médio (U$ K)'].unique())
dbf = dbf1.rename(columns={'Founded': 'Ano de fundação', 'Rating':'Nota','Sector':'Setor', 'Revenue': 'Faturamento (U$ MM)'
    , 'Type of ownership':'Tipo de organização'})
dbf['Tipo de organização']=dbf['Tipo de organização'].replace('Nonprofit Organization', 'Sem fins lucrativos').replace('Company - Public',
                                'Empresa pública').replace('Private Practice / Firm', 'Consultório privado / Firma').replace('Company - Private',
                                'Empresa privada').replace('Government', 'Governamental').replace('Other Organization', 'Sem informação').replace('Self-employed',
                                'Autônomo').replace('College / University', 'Faculdade / Universidade').replace('Contract', 'Contrato').replace('Subsidiary or Business Segment',
                                'Segmento de negócio ou empresarial')
dbf['Setor'] = dbf['Setor'].replace('Insurance', 'Seguros').replace('Business Services', 'Serviços de Negócios').replace('Manufacturing'
                                     , 'Indústria' ).replace('Information Technology', 'Tecnologia da Informação').replace('Retail', 'Varejista').replace('Government',
                                     'Governamental').replace('Finance', 'Mercado Financeiro').replace('Health Care', 'Saúde').replace('Media', 'Mídia').replace('Education',
                                     'Educação').replace('Consumer Services', 'Serviços ao consumidor').replace('Non-Profit', 'Sem fins lucrativos').replace('Accounting & Legal',
                                     'Advocacia e Contabilidade').replace('Agriculture & Forestry', 'Agricultura').replace('Construction, Repair & Maintenance'
                                     , 'Construção e Reparos').replace('Travel & Tourism', 'Viagem e Turismo').replace('Real Estate'
                                     , 'Imobiliário').replace('Telecommunications', 'Telecomunicações').replace('Transportation & Logistics',
                                     'Transporte e Logística').replace('Aerospace & Defense', 'Aeroespacial e Defesa').replace('Oil, Gas, Energy & Utilities',
                                     'Óleo, Gás, Energia e Utilidades').replace('Biotech & Pharmaceuticals', 'Farmacêutico e Biotecnologia')

#st.write(dbf['Ano de fundação'].value_counts())
#st.write(dbf['Salário Médio (U$ K)'].value_counts())
#st.dataframe(dbf)

erros_mae = [27.14, 26.86, 26.57, 26.25, 26.62, 27.52, 26.65, 26.75, 26.95, 27.04]

erros_mae_media = round(sum(erros_mae)/len(erros_mae),1)
#st.write(erros_mae_media)

erros_mae2 = [26.91, 26.91, 26.99, 26.97, 26.97, 27.09, 27.13, 26.95, 27.23, 30.37]

erros_melhor_arvore = 27.05 #depth 4
erros_melhor_floresta = 26.93 #depth 4 estimators 100




df_numeric = dbf.select_dtypes(include=['float64', 'int64'])

# Calcula a matriz de correlação
correlation_matrix = df_numeric.corr().round(2)

# Gera o gráfico interativo com Plotly
fig = px.imshow(
    correlation_matrix,
    text_auto=True,
    color_continuous_scale='Viridis',
    zmin=-1, zmax=1,
    title="Matriz de Correlação - Variáveis Numéricas"
)

fig.update_layout(width=800, height=800)
dbf2 = dbf.drop(columns=['Engenheiro de Dados','Número de Competidores Listados', 'Cargo Senior'])
df_numeric2 = dbf2.select_dtypes(include=['float64', 'int64'])

# Calcula a matriz de correlação
correlation_matrix2 = df_numeric2.corr().round(2)

# Gera o gráfico interativo com Plotly
fig2 = px.imshow(
    correlation_matrix2,
    text_auto=True,
    color_continuous_scale='Viridis',
    zmin=-1, zmax=1,
    title="Matriz de Correlação - Variáveis Numéricas"
)

fig2.update_layout(width=800, height=800)

st.subheader("Previsão de Salários de Vagas relacionadas a Ciência de Dados no Glassdoor")


tab1, tab2, tab3, tab4= st.tabs(["O problema e O objetivo",
                                        "O banco de dados","Aprendizado de máquina", "Conclusões"])


with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.write('O Glassdoor é um famoso site de busca de vagas de emprego, que permite a possibilidade de ver avaliações de outras pessoas que já trabalharam na empresa.')
        st.write('Neste problema, temos um banco de dados com vagas relacionadas a Ciência de Dados, com diversas informações que vão desde o salário estimado da vaga ao local da sede da empresa.')
        st.write('Pensando em que estratégia uma pessoa da área poderia adotar para escolher um emprego bem pago, realizamos os pré-processamentos descritos no relatório enviado em conjunto a este dashboard e a aplicação de uma Árvore de Decisão para identificar quais características que podemos extrair deste banco de dados são as mais significativas para a determinação da média do salário estimado de uma vaga em Ciência de Dados.')
    with col2:
        st.image(r"C:\Users\55859\Desktop\7_semestre\MS905\projetofinal\glassdoor.png")


with tab2:

    st.write('O banco de dados após o processamento é dado pela tabela abaixo. Em seguida, podemos ver a matriz de correlação das suas colunas numéricas.')
    st.dataframe(dbf)

    st.plotly_chart(fig)
    st.write('Observando a matriz de correlação, podemos eliminar do nosso aprendizado de máquina as seguintes colunas: Número de Competidores Listados e Cargo Senior, por terem correlação 0 com a coluna de Salário Médio (U$ K); Engenheiro de Dados, por estar contido na coluna Engenheiro e ter alta correlação com esta coluna.')
    st.write('A matriz de correlação resultante é:')
    st.plotly_chart(fig2)

with tab3:
    st.write('Com as colunas restantes, implementamos Árvores de Decisão, onde podemos variar o parâmetro da profundidade, de forma a achar a mais adequada para o problema. Escolhemos usar a profundidade de 6 camadas, pois foi a que gerou menor erro médio absoluto dos valores de 1 a 10. Como em todo aprendizado de máquinas, o banco original foi dividido em conjunto de treino (80%) e teste (20%).')
    st.write('Uma visualização da Árvore escolhida pode ser vista na seguinte imagem:')
    st.image(r"C:\Users\55859\Desktop\7_semestre\MS905\projetofinal\arvorefinaldevdd.png")
    st.write('Como escolhemos não reescalar as colunas numéricas, podemos mais facilmente identificar quais critérios levaram às decisões da árvore. Além disso, seu erro absoluto médio é dado por MAE = 26,74, ou seja, em média erramos por volta de 27 mil doláres na previsão do salário anual referente a uma dada vaga do conjunto de teste.')

with tab4:
    st.write('1. Como o banco de dados original trata majoritariamente de vagas relacionadas a Ciência de Dados com localização nos Estados Unidos, dados novos que desviem disso não serão bem previstos pelo modelo treinado. Ou seja, para vagas no Brasil, por exemplo, precisaríamos treinar o modelo de novo.')
    st.write('2. O erro médio absoluto na previsão do salário anual médio foi de 26,74 mil doláres, representando 61,5% do menor valor de salário médio no banco de dados e 9,8% do maior valor. É um erro grande, principalmente para os salários mais baixos. Esse erro deve poder ser mitigado ao obter mais amostras para esse banco de dados.')
    st.write('3. Apesar disso, para o objetivo de montar uma estratégia que ajude uma pessoa a escolher que características procurar numa vaga para que sua expectativa de salário seja atingida, o modelo de Árvore de Decisão é excelente, pois mostra exatamente quais características foram as dominantes na previsão do salário.')

#streamlit run C:\Users\55859\PycharmProjects\PythonProject\projetofinal.py