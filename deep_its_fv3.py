#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import boxcox


# # Deep Learning -ITS Policies Impact level in B5G slices

# ## Data importing and pre-processing

# In[2]:


# Dados referentes a App S
df_s_fs = pd.read_csv(r"C:\Users\sarai\Documents\doutorado\artigos\producao\vtc2024\df_s_fs.csv")
df_s_fq = pd.read_csv(r"C:\Users\sarai\Documents\doutorado\artigos\producao\vtc2024\df_s_fq.csv")
df_s_fn = pd.read_csv(r"C:\Users\sarai\Documents\doutorado\artigos\producao\vtc2024\df_s_fn.csv")
df_s_fs['pdr'] = df_s_fs.rec_serv_s_fs / df_s_fs.env_car_s_fs
df_s_fq['pdr'] = df_s_fq.rec_serv_s_fq / df_s_fq.env_car_s_fq
df_s_fn['pdr'] = df_s_fn.rec_serv_s_fn / df_s_fn.env_car_s_fn
df_s_fs.rename(columns={"time1_size_s_fs": "time", "rec_serv_s_fs": "rec_serv", "env_car_s_fs": "env_car", "time2_delay_s_fs": "time2_delay", "delay_rtt_s_fs": "rtt", "ncars_s_fs": "ncars"}, inplace=True)
df_s_fq.rename(columns={"time1_size_s_fq": "time", "rec_serv_s_fq": "rec_serv", "env_car_s_fq": "env_car", "time2_delay_s_fq": "time2_delay", "delay_rtt_s_fq": "rtt", "ncars_s_fq": "ncars"}, inplace=True)
df_s_fn.rename(columns={"time1_size_s_fn": "time", "rec_serv_s_fn": "rec_serv", "env_car_s_fn": "env_car", "time2_delay_s_fn": "time2_delay", "delay_rtt_s_fn": "rtt", "ncars_s_fn": "ncars"}, inplace=True)
df_s_fs.drop(columns=['time2_delay'], inplace=True)
df_s_fq.drop(columns=['time2_delay'], inplace=True)
df_s_fn.drop(columns=['time2_delay'], inplace=True)
df_s_fs['app'] = 'S' # Safety
df_s_fq['app'] = 'S' # Safety
df_s_fn['app'] = 'S' # Safety
df_s_fs['cat_req'] = 'mbclcp' #medium bandwidth / critical latency / critical priority
df_s_fq['cat_req'] = 'mbclcp' #medium bandwidth / critical latency / critical priority
df_s_fn['cat_req'] = 'mbclcp' #medium bandwidth / critical latency / critical priority
df_s_fs['approach'] = 'FS'
df_s_fq['approach'] = 'FQ'
df_s_fn['approach'] = 'FN'
df_s = pd.concat([df_s_fs, df_s_fq, df_s_fn])

# Dados referentes a App E
df_e_fs = pd.read_csv(r"C:\Users\sarai\Documents\doutorado\artigos\producao\vtc2024\df_e_fs.csv")
df_e_fq = pd.read_csv(r"C:\Users\sarai\Documents\doutorado\artigos\producao\vtc2024\df_e_fq.csv")
df_e_fn = pd.read_csv(r"C:\Users\sarai\Documents\doutorado\artigos\producao\vtc2024\df_e_fn.csv")
df_e_fs['pdr'] = df_e_fs.rec_serv_e_fs / df_e_fs.env_car_e_fs
df_e_fq['pdr'] = df_e_fq.rec_serv_e_fq / df_e_fq.env_car_e_fq
df_e_fn['pdr'] = df_e_fn.rec_serv_e_fn / df_e_fn.env_car_e_fn
df_e_fs.rename(columns={"time1_size_e_fs": "time", "rec_serv_e_fs": "rec_serv", "env_car_e_fs": "env_car", "time2_delay_e_fs": "time2_delay", "delay_rtt_e_fs": "rtt", "ncars_e_fs": "ncars"}, inplace=True)
df_e_fq.rename(columns={"time1_size_e_fq": "time", "rec_serv_e_fq": "rec_serv", "env_car_e_fq": "env_car", "time2_delay_e_fq": "time2_delay", "delay_rtt_e_fq": "rtt", "ncars_e_fq": "ncars"}, inplace=True)
df_e_fn.rename(columns={"time1_size_e_fn": "time", "rec_serv_e_fn": "rec_serv", "env_car_e_fn": "env_car", "time2_delay_e_fn": "time2_delay", "delay_rtt_e_fn": "rtt", "ncars_e_fn": "ncars"}, inplace=True)
df_e_fs.drop(columns=['time2_delay'], inplace=True)
df_e_fq.drop(columns=['time2_delay'], inplace=True)
df_e_fn.drop(columns=['time2_delay'], inplace=True)
df_e_fs['app'] = 'E' # Efficiency
df_e_fq['app'] = 'E' # Efficiency
df_e_fn['app'] = 'E' # Efficiency
df_e_fs['cat_req'] = 'mbmlhp' #medium bandwidth / medium latency / high priority
df_e_fq['cat_req'] = 'mbmlhp' #medium bandwidth / medium latency / high priority
df_e_fn['cat_req'] = 'mbmlhp' #medium bandwidth / medium latency / high priority
df_e_fs['approach'] = 'FS'
df_e_fq['approach'] = 'FQ'
df_e_fn['approach'] = 'FN'
df_e = pd.concat([df_e_fs, df_e_fq, df_e_fn])

# Dados referentes a App E2
df_e2_fs = pd.read_csv(r"C:\Users\sarai\Documents\doutorado\artigos\producao\vtc2024\df_e2_fs.csv")
df_e2_fq = pd.read_csv(r"C:\Users\sarai\Documents\doutorado\artigos\producao\vtc2024\df_e2_fq.csv")
df_e2_fn = pd.read_csv(r"C:\Users\sarai\Documents\doutorado\artigos\producao\vtc2024\df_e2_fn.csv")
df_e2_fs['pdr'] = df_e2_fs.rec_serv_e2_fs / df_e2_fs.env_car_e2_fs
df_e2_fq['pdr'] = df_e2_fq.rec_serv_e2_fq / df_e2_fq.env_car_e2_fq
df_e2_fn['pdr'] = df_e2_fn.rec_serv_e2_fn / df_e2_fn.env_car_e2_fn
df_e2_fs.rename(columns={"time1_size_e2_fs": "time", "rec_serv_e2_fs": "rec_serv", "env_car_e2_fs": "env_car", "time2_delay_e2_fs": "time2_delay", "delay_rtt_e2_fs": "rtt", "ncars_e2_fs": "ncars"}, inplace=True)
df_e2_fq.rename(columns={"time1_size_e2_fq": "time", "rec_serv_e2_fq": "rec_serv", "env_car_e2_fq": "env_car", "time2_delay_e2_fq": "time2_delay", "delay_rtt_e2_fq": "rtt", "ncars_e2_fq": "ncars"}, inplace=True)
df_e2_fn.rename(columns={"time1_size_e2_fn": "time", "rec_serv_e2_fn": "rec_serv", "env_car_e2_fn": "env_car", "time2_delay_e2_fn": "time2_delay", "delay_rtt_e2_fn": "rtt", "ncars_e2_fn": "ncars"}, inplace=True)
df_e2_fs.drop(columns=['time2_delay'], inplace=True)
df_e2_fq.drop(columns=['time2_delay'], inplace=True)
df_e2_fn.drop(columns=['time2_delay'], inplace=True)
df_e2_fs['app'] = 'E2'
df_e2_fq['app'] = 'E2'
df_e2_fn['app'] = 'E2'
df_e2_fs['cat_req'] = 'hbmlmp' #high bandwidth / medium latency / medium priority
df_e2_fq['cat_req'] = 'hbmlmp' #high bandwidth / medium latency / medium priority
df_e2_fn['cat_req'] = 'hbmlmp' #high bandwidth / medium latency / medium priority
df_e2_fs['approach'] = 'FS'
df_e2_fq['approach'] = 'FQ'
df_e2_fn['approach'] = 'FN'
df_e2 = pd.concat([df_e2_fs, df_e2_fq, df_e2_fn])

# Dados referentes a App G
df_g_fs = pd.read_csv(r"C:\Users\sarai\Documents\doutorado\artigos\producao\vtc2024\df_g_fs.csv")
df_g_fq = pd.read_csv(r"C:\Users\sarai\Documents\doutorado\artigos\producao\vtc2024\df_g_fq.csv")
df_g_fn = pd.read_csv(r"C:\Users\sarai\Documents\doutorado\artigos\producao\vtc2024\df_g_fn.csv")
df_g_fs['pdr'] = df_g_fs.rec_serv_g_fs / df_g_fs.env_car_g_fs
df_g_fq['pdr'] = df_g_fq.rec_serv_g_fq / df_g_fq.env_car_g_fq
df_g_fn['pdr'] = df_g_fn.rec_serv_g_fn / df_g_fn.env_car_g_fn
df_g_fs.rename(columns={"time1_size_g_fs": "time", "rec_serv_g_fs": "rec_serv", "env_car_g_fs": "env_car", "time2_delay_g_fs": "time2_delay", "delay_rtt_g_fs": "rtt", "ncars_g_fs": "ncars"}, inplace=True)
df_g_fq.rename(columns={"time1_size_g_fq": "time", "rec_serv_g_fq": "rec_serv", "env_car_g_fq": "env_car", "time2_delay_g_fq": "time2_delay", "delay_rtt_g_fq": "rtt", "ncars_g_fq": "ncars"}, inplace=True)
df_g_fn.rename(columns={"time1_size_g_fn": "time", "rec_serv_g_fn": "rec_serv", "env_car_g_fn": "env_car", "time2_delay_g_fn": "time2_delay", "delay_rtt_g_fn": "rtt", "ncars_g_fn": "ncars"}, inplace=True)
df_g_fs.drop(columns=['time2_delay'], inplace=True)
df_g_fq.drop(columns=['time2_delay'], inplace=True)
df_g_fn.drop(columns=['time2_delay'], inplace=True)
df_g_fs['app'] = 'G'
df_g_fq['app'] = 'G'
df_g_fn['app'] = 'G'
df_g_fs['cat_req'] = 'lblllp' #low bandwidth / low latency / low priority
df_g_fq['cat_req'] = 'lblllp' #low bandwidth / low latency / low priority
df_g_fn['cat_req'] = 'lblllp' #low bandwidth / low latency / low priority
df_g_fs['approach'] = 'FS'
df_g_fq['approach'] = 'FQ'
df_g_fn['approach'] = 'FN'
df_g = pd.concat([df_g_fs, df_g_fq, df_g_fn])
df_g.dropna(inplace=True) # elimina registros nulos


# ## Feature Engineering

# In[3]:


############### Engenharia de features ########################################

# substitui valores nulos/zero por epsilon (1e-6), assegurando que todos os valores de RTT sejam positivos p/ boxcox
# clip(lower=1e-6) troca apenas zeros eventuais — não desloca a média.
# boxcox devolve a série transformada e o λ ótimo; guardar o λ permite reverter ou aplicar em novos dados
def apply_boxcox(df, col_in='rtt', col_out='bc_rtt'):
    # 1) garante valores positivos
    rtt_pos = df[col_in].clip(lower=1e-6)
    # 2) aplica Box-Cox
    df[col_out], lam = boxcox(rtt_pos)
    return lam
λ_e  = apply_boxcox(df_e)
λ_e2 = apply_boxcox(df_e2)
λ_g  = apply_boxcox(df_g)
λ_s  = apply_boxcox(df_s)
print(f"λs escolhidos no boxcox: E={λ_e:.3f}, E2={λ_e2:.3f}, G={λ_g:.3f}, S={λ_s:.3f}")

# Taxa de mudança para RTT
df_e['rtt_change'] = df_e['rtt'].diff() / df_e['time'].diff()
df_e2['rtt_change'] = df_e2['rtt'].diff() / df_e2['time'].diff()
df_g['rtt_change'] = df_g['rtt'].diff() / df_g['time'].diff()
df_s['rtt_change'] = df_s['rtt'].diff() / df_s['time'].diff()
# Taxa de mudança para PDR
df_e['pdr_change'] = df_e['pdr'].diff() / df_e['time'].diff()
df_e2['pdr_change'] = df_e2['pdr'].diff() / df_e2['time'].diff()
df_g['pdr_change'] = df_g['pdr'].diff() / df_g['time'].diff()
df_s['pdr_change'] = df_s['pdr'].diff() / df_s['time'].diff()

# Agregações temporais: Média móvel de 3 períodos para RTT
df_e['rtt_mam'] = df_e['rtt'].rolling(window=3).mean()
df_e2['rtt_mam'] = df_e2['rtt'].rolling(window=3).mean()
df_g['rtt_mam'] = df_g['rtt'].rolling(window=3).mean()
df_s['rtt_mam'] = df_s['rtt'].rolling(window=3).mean()
# Agregações temporais: Média móvel de 3 períodos para PDR
df_e['pdr_mam'] = df_e['pdr'].rolling(window=3).mean()
df_e2['pdr_mam'] = df_e2['pdr'].rolling(window=3).mean()
df_g['pdr_mam'] = df_g['pdr'].rolling(window=3).mean()
df_s['pdr_mam'] = df_s['pdr'].rolling(window=3).mean()

# Agregações temporais: Desvio padrao de 3 períodos para RTT
df_e['rtt_masd'] = df_e['rtt'].rolling(window=3).std()
df_e2['rtt_masd'] = df_e2['rtt'].rolling(window=3).std()
df_g['rtt_masd'] = df_g['rtt'].rolling(window=3).std()
df_s['rtt_masd'] = df_s['rtt'].rolling(window=3).std()
# Agregações temporais: Desvio padrao de 3 períodos para PDR
df_e['pdr_masd'] = df_e['pdr'].rolling(window=3).std()
df_e2['pdr_masd'] = df_e2['pdr'].rolling(window=3).std()
df_g['pdr_masd'] = df_g['pdr'].rolling(window=3).std()
df_s['pdr_masd'] = df_s['pdr'].rolling(window=3).std()

# Transformações logarítmicas para o RTT (natural)
df_e['log_rtt'] = np.log(df_e['rtt'])
df_e2['log_rtt'] = np.log(df_e2['rtt'])
df_g['log_rtt'] = np.log(df_g['rtt'])
df_s['log_rtt'] = np.log(df_s['rtt'])
# Transformações logarítmicas para o PDR (natural)
df_e['log_pdr'] = np.log(df_e['pdr'])
df_e2['log_pdr'] = np.log(df_e2['pdr'])
df_g['log_pdr'] = np.log(df_g['pdr'])
df_s['log_pdr'] = np.log(df_s['pdr'])

# Transformações logarítmicas para o RTT (base2)
df_e['log2_rtt'] = np.log2(df_e['rtt'])
df_e2['log2_rtt'] = np.log2(df_e2['rtt'])
df_g['log2_rtt'] = np.log2(df_g['rtt'])
df_s['log2_rtt'] = np.log2(df_s['rtt'])
# Transformações logarítmicas para o PDR (base2)
df_e['log2_pdr'] = np.log2(df_e['pdr'])
df_e2['log2_pdr'] = np.log2(df_e2['pdr'])
df_g['log2_pdr'] = np.log2(df_g['pdr'])
df_s['log2_pdr'] = np.log2(df_s['pdr'])

# Transformações polinomiais para o RTT (elevando ao quadrado)
df_e['rtt_sqrd'] = df_e['rtt'] ** 2
df_e2['rtt_sqrd'] = df_e2['rtt'] ** 2
df_g['rtt_sqrd'] = df_g['rtt'] ** 2
df_s['rtt_sqrd'] = df_s['rtt'] ** 2
# Transformações polinomiais para o PDR (elevando ao quadrado)
df_e['pdr_sqrd'] = df_e['pdr'] ** 2
df_e2['pdr_sqrd'] = df_e2['pdr'] ** 2
df_g['pdr_sqrd'] = df_g['pdr'] ** 2
df_s['pdr_sqrd'] = df_s['pdr'] ** 2

# unindo todo o dataset, considerando todas aplicaçoes e abordagens avaliadas em simulacao - excluindo nulos
df_sim = pd.concat([df_e, df_e2, df_s, df_g])
df_sim.dropna(inplace=True)
#Removendo coluna de tempo
#df_sim.drop(columns=['time'], inplace=True)

# Campo para nivel de impacto
df_sim['impact_level'] = 'undefined'

# RTT utilizado de referencia
rtt = 'bc_rtt'


# In[4]:


#sample dos dados
df_sim.sample(5)


# In[5]:


# distribuição dos dados no dataframe
df_sim.describe(include="all")


# ## Dealing with categorical values

# In[6]:


from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder

# tratando colunas categoricas
colunas_categoricas = ['app', 'approach', 'cat_req']

one_hot_enc = make_column_transformer(
    (OneHotEncoder(handle_unknown = 'ignore'),
    colunas_categoricas),
    remainder='passthrough')

df_sim_ohe = one_hot_enc.fit_transform(df_sim)
df_sim_ohe = pd.DataFrame(df_sim_ohe, columns=one_hot_enc.get_feature_names_out())
df_sim_ohe.columns = [col.replace('remainder__', '') for col in df_sim_ohe.columns]
df_sim_ohe.columns = [col.replace('onehotencoder__', 'ohe__') for col in df_sim_ohe.columns]
df_sim_ohe.head()


# In[7]:


# Formato final do dataframe
df_sim_ohe.shape


# In[8]:


# Identificando colunas
df_sim_ohe.columns


# ## Applications analisys and impact level attribution

# ### App E (Efficiency 500Kbps - Priority 1)

# In[9]:


import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Criando figura com subplots - App E (Eficiencia 500Kbps - Prioridade 1)
fig = make_subplots(rows=2, cols=3, subplot_titles=('Framework', 'QoS only', 'RMSDVN'),
                    specs=[[{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}], [{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}]])

# Filtrando App E
dfse = df_sim_ohe.query('`ohe__app_E` == 1 and `ohe__approach_FS` == 1')
dfqe = df_sim_ohe.query('`ohe__app_E` == 1 and `ohe__approach_FQ` == 1')
dfne = df_sim_ohe.query('`ohe__app_E` == 1 and `ohe__approach_FN` == 1')

# Adicionando dados aos subplots
fig.add_trace(go.Scatter(x=dfse['time'], y=dfse[rtt], mode='lines', name='RTT', line=dict(color='orange', width=2)), row=1, col=1, secondary_y=True)
fig.add_trace(go.Scatter(x=dfse['time'], y=dfse['env_car']/1000, mode='lines', name='Car Sent', line=dict(color='blue', dash='dot')), row=1, col=1, secondary_y=False)
fig.add_trace(go.Scatter(x=dfse['time'], y=dfse['rec_serv']/1000, mode='lines', name='Car Received', line=dict(color='red', dash='dash')), row=1, col=1, secondary_y=False)

fig.add_trace(go.Scatter(x=dfqe['time'], y=dfqe[rtt], mode='lines', name='RTT', line=dict(color='orange', width=2)), row=1, col=2, secondary_y=True)
fig.add_trace(go.Scatter(x=dfqe['time'], y=dfqe['env_car']/1000, mode='lines', name='Car Sent', line=dict(color='blue', dash='dot')), row=1, col=2, secondary_y=False)
fig.add_trace(go.Scatter(x=dfqe['time'], y=dfqe['rec_serv']/1000, mode='lines', name='Car Received', line=dict(color='red', dash='dash')), row=1, col=2, secondary_y=False)

fig.add_trace(go.Scatter(x=dfne['time'], y=dfne[rtt], mode='lines', name='RTT', line=dict(color='orange', width=2)), row=1, col=3, secondary_y=True)
fig.add_trace(go.Scatter(x=dfne['time'], y=dfne['env_car']/1000, mode='lines', name='Car Sent', line=dict(color='blue', dash='dot')), row=1, col=3, secondary_y=False)
fig.add_trace(go.Scatter(x=dfne['time'], y=dfne['rec_serv']/1000, mode='lines', name='Car Received', line=dict(color='red', dash='dash')), row=1, col=3, secondary_y=False)

fig.add_trace(go.Scatter(x=dfse['time'], y=dfse['pdr'], mode='lines', name='PDR FS', line=dict(color='green', dash='dot', width=3)), row=2, col=1, secondary_y=False)
fig.add_trace(go.Scatter(x=dfqe['time'], y=dfqe['pdr'], mode='lines', name='PDR FQ', line=dict(color='green', dash='dot', width=3)), row=2, col=2, secondary_y=False)
fig.add_trace(go.Scatter(x=dfne['time'], y=dfne['pdr'], mode='lines', name='PDR FN', line=dict(color='green', dash='dot', width=3)), row=2, col=3, secondary_y=False)

# Atualizando layout para um design coerente
fig.update_layout(
    title='Comparative Performance Analysis - App E (Eficiencia 500Kbps - Prioridade 1)',
    xaxis_title='Time (s)',
    legend_title='Metrics',
    template='plotly_dark',
    height=700,  # Ajustar a altura se necessário
    #width=3700   # Ajustar a largura para caber os três plots
)

# Atualizando configurações dos eixos y
for i in range(1, 4):
    fig.update_yaxes(title_text="Throughput (Kbps)", range=[0, 700], row=1, col=i, secondary_y=False)
    fig.update_yaxes(title_text="Packet Delivery Rate (%)", row=2, col=i, secondary_y=False)
    fig.update_yaxes(title_text="RTT (ms)", range=[0, 400], row=1, col=i, secondary_y=True)  # Ajuste do range do eixo Y secundário

# Mostrando gráfico
fig.show()


# #### Define requirements, classify and analisys - App E (Efficiency 500Kbps - Priority 1)

# In[10]:


# Requisitos de impacto App E (Eficiencia 500Kbps - Prioridade 1)
low = ['<=200', '>=80', '>350']
medium = ['<=250', '>=50', '>=200']
high = ['>250', '<50', '<200']
#impact_level = ['low', 'medium', 'high']
imp_e = {'low': low, 'medium':medium, 'high':high}
imp_e = pd.DataFrame.from_dict(imp_e)
imp_e.index = ['RTT (ms)', 'PDR (%)', 'Rx (Kbps)']
print('\nRequisitos de impacto App E (Eficiencia 500Kbps - Prioridade 1)\n')
imp_e = pd.DataFrame.from_dict(imp_e)
display(imp_e)

# classificando App E
df_sim_ohe.loc[(df_sim_ohe['ohe__app_E'] == 1) & (df_sim_ohe[rtt] <= 200) & (df_sim_ohe['pdr'] >= 0.80) & (df_sim_ohe['rec_serv'] > 350000),'impact_level'] = 'low'
df_sim_ohe.loc[(df_sim_ohe['ohe__app_E'] == 1) & (df_sim_ohe[rtt] <= 250) & (df_sim_ohe['pdr'] >= 0.50) & (df_sim_ohe['rec_serv'] >= 200000) & (df_sim_ohe['impact_level'] != 'low'), 'impact_level'] = 'medium'
df_sim_ohe.loc[(df_sim_ohe['ohe__app_E'] == 1) & ((df_sim_ohe[rtt] > 250) | (df_sim_ohe['pdr'] < 0.50) | (df_sim_ohe['rec_serv'] < 200000)) & (df_sim_ohe['impact_level'] != 'medium'), 'impact_level'] = 'high'

# App E - analise impacto
imp_e_fs = pd.DataFrame(df_sim_ohe.query('`ohe__app_E` == 1 and `ohe__approach_FS` == 1').groupby(['impact_level']).count()['ohe__approach_FS'])
imp_e_fq = pd.DataFrame(df_sim_ohe.query('`ohe__app_E` == 1 and `ohe__approach_FQ` == 1').groupby(['impact_level']).count()['ohe__approach_FQ'])
imp_e_fn = pd.DataFrame(df_sim_ohe.query('`ohe__app_E` == 1 and `ohe__approach_FN` == 1').groupby(['impact_level']).count()['ohe__approach_FN'])
analise_e = imp_e_fs.join(imp_e_fq).join(imp_e_fn)
analise_e.columns = ['FS', 'FQ', 'FN']
print('\nAnalise de impacto App E (Eficiencia 500Kbps - Prioridade 1)\n')
display(analise_e.T.style.background_gradient())


# In[11]:


# Requisitos de impacto App G (Internet Generica - 500Kbps - Prioridade 3)
low = ['<=50', '>=50', '>250']
medium = ['<=100', '>=35', '>=100']
high = ['>100', '<35', '<100']
imp_g = {'low': low, 'medium':medium, 'high':high}
imp_g = pd.DataFrame.from_dict(imp_g)
imp_g.index = ['RTT (ms)', 'PDR (%)', 'Rx (Kbps)']
print('\nRequisitos de impacto App G (Internet Generica - 500Kbps - Prioridade 3)\n')
imp_g = pd.DataFrame.from_dict(imp_g)
display(imp_g)

# Classificando G
df_sim_ohe.loc[(df_sim_ohe['ohe__app_G'] == 1) & (df_sim_ohe[rtt] <= 50) & (df_sim_ohe['pdr'] >= 0.50) & (df_sim_ohe['rec_serv'] >= 250000),'impact_level'] = 'low'
df_sim_ohe.loc[(df_sim_ohe['ohe__app_G'] == 1) & (df_sim_ohe[rtt] <= 100) & (df_sim_ohe['pdr'] >= 0.35) & (df_sim_ohe['rec_serv'] >= 100000) & (df_sim_ohe['impact_level'] != 'low'), 'impact_level'] = 'medium'
df_sim_ohe.loc[(df_sim_ohe['ohe__app_G'] == 1) & ((df_sim_ohe[rtt] > 100) | (df_sim_ohe['pdr'] < 0.35) | (df_sim_ohe['rec_serv'] < 100000)) & (df_sim_ohe['impact_level'] != 'medium'), 'impact_level'] = 'high'

# App G - analise impacto
imp_g_fs = pd.DataFrame(df_sim_ohe.query('`ohe__app_G` == 1 and `ohe__approach_FS` == 1').groupby(['impact_level']).count()['ohe__approach_FS'])
imp_g_fq = pd.DataFrame(df_sim_ohe.query('`ohe__app_G` == 1 and `ohe__approach_FQ` == 1').groupby(['impact_level']).count()['ohe__approach_FQ'])
imp_g_fn = pd.DataFrame(df_sim_ohe.query('`ohe__app_G` == 1 and `ohe__approach_FN` == 1').groupby(['impact_level']).count()['ohe__approach_FN'])
analise_g = imp_g_fs.join(imp_g_fq).join(imp_g_fn)
analise_g.columns = ['FS', 'FQ', 'FN']
print('\nAnalise de impacto App G (Internet Generica - 500Kbps - Prioridade 3)\n')
display(analise_g.T.style.background_gradient())


# ### App E2 (Entretenimento 1Mbps - Prioridade 2)

# In[12]:


# Criando figura com subplots - App E2 (Entretenimento 1Mbps - Prioridade 2)
fig = make_subplots(rows=2, cols=3, subplot_titles=('Framework', 'QoS only', 'RMSDVN'),
                    specs=[[{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}], [{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}]])

# Filtrando App E2
dfse2 = df_sim_ohe.query('`ohe__app_E2` == 1 and `ohe__approach_FS` == 1')
dfqe2 = df_sim_ohe.query('`ohe__app_E2` == 1 and `ohe__approach_FQ` == 1')
dfne2 = df_sim_ohe.query('`ohe__app_E2` == 1 and `ohe__approach_FN` == 1')

# Adicionando dados aos subplots
fig.add_trace(go.Scatter(x=dfse2['time'], y=dfse2[rtt], mode='lines', name='RTT', line=dict(color='orange', width=2)), row=1, col=1, secondary_y=True)
fig.add_trace(go.Scatter(x=dfse2['time'], y=dfse2['env_car']/1000, mode='lines', name='Car Sent', line=dict(color='blue', dash='dot')), row=1, col=1, secondary_y=False)
fig.add_trace(go.Scatter(x=dfse2['time'], y=dfse2['rec_serv']/1000, mode='lines', name='Car Received', line=dict(color='red', dash='dash')), row=1, col=1, secondary_y=False)

fig.add_trace(go.Scatter(x=dfqe2['time'], y=dfqe2[rtt], mode='lines', name='RTT', line=dict(color='orange', width=2)), row=1, col=2, secondary_y=True)
fig.add_trace(go.Scatter(x=dfqe2['time'], y=dfqe2['env_car']/1000, mode='lines', name='Car Sent', line=dict(color='blue', dash='dot')), row=1, col=2, secondary_y=False)
fig.add_trace(go.Scatter(x=dfqe2['time'], y=dfqe2['rec_serv']/1000, mode='lines', name='Car Received', line=dict(color='red', dash='dash')), row=1, col=2, secondary_y=False)

fig.add_trace(go.Scatter(x=dfne2['time'], y=dfne2[rtt], mode='lines', name='RTT', line=dict(color='orange', width=2)), row=1, col=3, secondary_y=True)
fig.add_trace(go.Scatter(x=dfne2['time'], y=dfne2['env_car']/1000, mode='lines', name='Car Sent', line=dict(color='blue', dash='dot')), row=1, col=3, secondary_y=False)
fig.add_trace(go.Scatter(x=dfne2['time'], y=dfne2['rec_serv']/1000, mode='lines', name='Car Received', line=dict(color='red', dash='dash')), row=1, col=3, secondary_y=False)

fig.add_trace(go.Scatter(x=dfse2['time'], y=dfse2['pdr'], mode='lines', name='PDR FS', line=dict(color='green', dash='dot', width=3)), row=2, col=1, secondary_y=False)
fig.add_trace(go.Scatter(x=dfqe2['time'], y=dfqe2['pdr'], mode='lines', name='PDR FQ', line=dict(color='green', dash='dot', width=3)), row=2, col=2, secondary_y=False)
fig.add_trace(go.Scatter(x=dfne2['time'], y=dfne2['pdr'], mode='lines', name='PDR FN', line=dict(color='green', dash='dot', width=3)), row=2, col=3, secondary_y=False)

# Atualizando layout para um design coerente
fig.update_layout(
    title='Comparative Performance Analysis - App E2 (Entretenimento 1Mbps - Prioridade 2)',
    xaxis_title='Time (s)',
    legend_title='Metrics',
    template='plotly_dark',
    height=700,  # Ajustar a altura se necessário
    #width=3700   # Ajustar a largura para caber os três plots
)

# Atualizando configurações dos eixos y
for i in range(1, 4):
    fig.update_yaxes(title_text="Throughput (Kbps)", range=[0, 1400], row=1, col=i, secondary_y=False)
    fig.update_yaxes(title_text="Packet Delivery Rate (%)", row=2, col=i, secondary_y=False)
    fig.update_yaxes(title_text="RTT (ms)", range=[0, 400], row=1, col=i, secondary_y=True)  # Ajuste do range do eixo Y secundário

# Mostrando gráfico
fig.show()


# #### Define requisitos, classifica e analise impacto - App E2 (Entretenimento 1Mbps - Prioridade 2))

# In[13]:


# Requisitos de impacto App E2 (Entretenimento 1Mbps - Prioridade 2)
low = ['<=70', '>=70', '>700']
medium = ['<=100', '>=50', '>=500']
high = ['>100', '<50', '<500']
imp_e2 = {'low': low, 'medium':medium, 'high':high}
imp_e2 = pd.DataFrame.from_dict(imp_e)
imp_e2.index = ['RTT (ms)', 'PDR (%)', 'Rx (Kbps)']
print('\nRequisitos de impacto App E2 (Entretenimento 1Mbps - Prioridade 2)\n')
imp_e2 = pd.DataFrame.from_dict(imp_e2)
display(imp_e2)

# Classificando E2
df_sim_ohe.loc[(df_sim_ohe['ohe__app_E2'] == 1) & (df_sim_ohe[rtt] <= 70) & (df_sim_ohe['pdr'] >= 0.70) & (df_sim_ohe['rec_serv'] > 700000),'impact_level'] = 'low'
df_sim_ohe.loc[(df_sim_ohe['ohe__app_E2'] == 1) & (df_sim_ohe[rtt] <= 100) & (df_sim_ohe['pdr'] >= 0.50) & (df_sim_ohe['rec_serv'] >= 500000) & (df_sim_ohe['impact_level'] != 'low'), 'impact_level'] = 'medium'
df_sim_ohe.loc[(df_sim_ohe['ohe__app_E2'] == 1) & ((df_sim_ohe[rtt] > 100) | (df_sim_ohe['pdr'] < 0.50) | (df_sim_ohe['rec_serv'] < 500000)) & (df_sim_ohe['impact_level'] != 'medium'), 'impact_level'] = 'high'

# App E2 - analise impacto
imp_e2_fs = pd.DataFrame(df_sim_ohe.query('`ohe__app_E2` == 1 and `ohe__approach_FS` == 1').groupby(['impact_level']).count()['ohe__approach_FS'])
imp_e2_fq = pd.DataFrame(df_sim_ohe.query('`ohe__app_E2` == 1 and `ohe__approach_FQ` == 1').groupby(['impact_level']).count()['ohe__approach_FQ'])
imp_e2_fn = pd.DataFrame(df_sim_ohe.query('`ohe__app_E2` == 1 and `ohe__approach_FN` == 1').groupby(['impact_level']).count()['ohe__approach_FN'])
analise_e2 = imp_e2_fs.join(imp_e2_fq).join(imp_e2_fn)
analise_e2.columns = ['FS', 'FQ', 'FN']
print('\nAnalise de impacto App E2 (Entretenimento 1Mbps - Prioridade 2)\n')
display(analise_e2.T.style.background_gradient())


# ### App G (Internet Generica - 500Kbps - Prioridade 3)

# In[14]:


# Criando figura com subplots - App G (Internet Generica - 500Kbps - Prioridade 3)
fig = make_subplots(rows=2, cols=3, subplot_titles=('Framework', 'QoS only', 'RMSDVN'),
                    specs=[[{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}], [{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}]])

# Filtrando App G
dfsg = df_sim_ohe.query('`ohe__app_G` == 1 and `ohe__approach_FS` == 1')
dfqg = df_sim_ohe.query('`ohe__app_G` == 1 and `ohe__approach_FQ` == 1')
dfng = df_sim_ohe.query('`ohe__app_G` == 1 and `ohe__approach_FN` == 1')

# Adicionando dados aos subplots
fig.add_trace(go.Scatter(x=dfsg['time'], y=dfsg[rtt], mode='lines', name='RTT', line=dict(color='orange', width=2)), row=1, col=1, secondary_y=True)
fig.add_trace(go.Scatter(x=dfsg['time'], y=dfsg['env_car']/1000, mode='lines', name='Car Sent', line=dict(color='blue', dash='dot')), row=1, col=1, secondary_y=False)
fig.add_trace(go.Scatter(x=dfsg['time'], y=dfsg['rec_serv']/1000, mode='lines', name='Car Received', line=dict(color='red', dash='dash')), row=1, col=1, secondary_y=False)

fig.add_trace(go.Scatter(x=dfqg['time'], y=dfqg[rtt], mode='lines', name='RTT', line=dict(color='orange', width=2)), row=1, col=2, secondary_y=True)
fig.add_trace(go.Scatter(x=dfqg['time'], y=dfqg['env_car']/1000, mode='lines', name='Car Sent', line=dict(color='blue', dash='dot')), row=1, col=2, secondary_y=False)
fig.add_trace(go.Scatter(x=dfqg['time'], y=dfqg['rec_serv']/1000, mode='lines', name='Car Received', line=dict(color='red', dash='dash')), row=1, col=2, secondary_y=False)

fig.add_trace(go.Scatter(x=dfng['time'], y=dfng[rtt], mode='lines', name='RTT', line=dict(color='orange', width=2)), row=1, col=3, secondary_y=True)
fig.add_trace(go.Scatter(x=dfng['time'], y=dfng['env_car']/1000, mode='lines', name='Car Sent', line=dict(color='blue', dash='dot')), row=1, col=3, secondary_y=False)
fig.add_trace(go.Scatter(x=dfng['time'], y=dfng['rec_serv']/1000, mode='lines', name='Car Received', line=dict(color='red', dash='dash')), row=1, col=3, secondary_y=False)

fig.add_trace(go.Scatter(x=dfsg['time'], y=dfsg['pdr'], mode='lines', name='PDR FS', line=dict(color='green', dash='dot', width=3)), row=2, col=1, secondary_y=False)
fig.add_trace(go.Scatter(x=dfqg['time'], y=dfqg['pdr'], mode='lines', name='PDR FQ', line=dict(color='green', dash='dot', width=3)), row=2, col=2, secondary_y=False)
fig.add_trace(go.Scatter(x=dfng['time'], y=dfng['pdr'], mode='lines', name='PDR FN', line=dict(color='green', dash='dot', width=3)), row=2, col=3, secondary_y=False)

# Atualizando layout para um design coerente
fig.update_layout(
    title='Comparative Performance Analysis - App G (Internet Generica - 500Kbps - Prioridade 3)',
    xaxis_title='Time (s)',
    legend_title='Metrics',
    template='plotly_dark',
    height=700,  # Ajustar a altura se necessário
    #width=3700   # Ajustar a largura para caber os três plots
)

# Atualizando configurações dos eixos y
for i in range(1, 4):
    fig.update_yaxes(title_text="Throughput (Kbps)", range=[0, 600], row=1, col=i, secondary_y=False)
    fig.update_yaxes(title_text="Packet Delivery Rate (%)", row=2, col=i, secondary_y=False)
    fig.update_yaxes(title_text="RTT (ms)", range=[0, 400], row=1, col=i, secondary_y=True)  # Ajuste do range do eixo Y secundário

# Mostrando gráfico
fig.show()


# #### Define requisitos, classifica e analise impacto - App G (Internet Generica - 500Kbps - Prioridade 3)

# In[15]:


# Requisitos de impacto App G (Internet Generica - 500Kbps - Prioridade 3)
low = ['<=50', '>=50', '>250']
medium = ['<=100', '>=35', '>=100']
high = ['>100', '<35', '<100']
imp_g = {'low': low, 'medium':medium, 'high':high}
imp_g = pd.DataFrame.from_dict(imp_g)
imp_g.index = ['RTT (ms)', 'PDR (%)', 'Rx (Kbps)']
print('\nRequisitos de impacto App G (Internet Generica - 500Kbps - Prioridade 3)\n')
imp_g = pd.DataFrame.from_dict(imp_g)
display(imp_g)

# Classificando G
df_sim_ohe.loc[(df_sim_ohe['ohe__app_G'] == 1) & (df_sim_ohe[rtt] <= 50) & (df_sim_ohe['pdr'] >= 0.50) & (df_sim_ohe['rec_serv'] >= 250000),'impact_level'] = 'low'
df_sim_ohe.loc[(df_sim_ohe['ohe__app_G'] == 1) & (df_sim_ohe[rtt] <= 100) & (df_sim_ohe['pdr'] >= 0.35) & (df_sim_ohe['rec_serv'] >= 100000) & (df_sim_ohe['impact_level'] != 'low'), 'impact_level'] = 'medium'
df_sim_ohe.loc[(df_sim_ohe['ohe__app_G'] == 1) & ((df_sim_ohe[rtt] > 100) | (df_sim_ohe['pdr'] < 0.35) | (df_sim_ohe['rec_serv'] < 100000)) & (df_sim_ohe['impact_level'] != 'medium'), 'impact_level'] = 'high'

# App G - analise impacto
imp_g_fs = pd.DataFrame(df_sim_ohe.query('`ohe__app_G` == 1 and `ohe__approach_FS` == 1').groupby(['impact_level']).count()['ohe__approach_FS'])
imp_g_fq = pd.DataFrame(df_sim_ohe.query('`ohe__app_G` == 1 and `ohe__approach_FQ` == 1').groupby(['impact_level']).count()['ohe__approach_FQ'])
imp_g_fn = pd.DataFrame(df_sim_ohe.query('`ohe__app_G` == 1 and `ohe__approach_FN` == 1').groupby(['impact_level']).count()['ohe__approach_FN'])
analise_g = imp_g_fs.join(imp_g_fq).join(imp_g_fn)
analise_g.columns = ['FS', 'FQ', 'FN']
print('\nAnalise de impacto App G (Internet Generica - 500Kbps - Prioridade 3)\n')
display(analise_g.T.style.background_gradient())


# ### App S (Safety/MEC - 500Kbps - Priority 0)

# In[16]:


# Criando figura com subplots - App S (Safety/MEC - 500Kbps - Prioridade 0)
fig = make_subplots(rows=2, cols=3, subplot_titles=('Framework', 'QoS only', 'RMSDVN'),
                    specs=[[{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}], [{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}]])

# Filtrando App S
dfss = df_sim_ohe.query('`ohe__app_S` == 1 and `ohe__approach_FS` == 1')
dfqs = df_sim_ohe.query('`ohe__app_S` == 1 and `ohe__approach_FQ` == 1')
dfns = df_sim_ohe.query('`ohe__app_S` == 1 and `ohe__approach_FN` == 1')

# Adicionando dados aos subplots
fig.add_trace(go.Scatter(x=dfss['time'], y=dfss[rtt], mode='lines', name='RTT', line=dict(color='orange', width=2)), row=1, col=1, secondary_y=True)
fig.add_trace(go.Scatter(x=dfss['time'], y=dfss['env_car']/1000, mode='lines', name='Car Sent', line=dict(color='blue', dash='dot')), row=1, col=1, secondary_y=False)
fig.add_trace(go.Scatter(x=dfss['time'], y=dfss['rec_serv']/1000, mode='lines', name='Car Received', line=dict(color='red', dash='dash')), row=1, col=1, secondary_y=False)

fig.add_trace(go.Scatter(x=dfqs['time'], y=dfqs[rtt], mode='lines', name='RTT', line=dict(color='orange', width=2)), row=1, col=2, secondary_y=True)
fig.add_trace(go.Scatter(x=dfqs['time'], y=dfqs['env_car']/1000, mode='lines', name='Car Sent', line=dict(color='blue', dash='dot')), row=1, col=2, secondary_y=False)
fig.add_trace(go.Scatter(x=dfqs['time'], y=dfqs['rec_serv']/1000, mode='lines', name='Car Received', line=dict(color='red', dash='dash')), row=1, col=2, secondary_y=False)

fig.add_trace(go.Scatter(x=dfns['time'], y=dfns[rtt], mode='lines', name='RTT', line=dict(color='orange', width=2)), row=1, col=3, secondary_y=True)
fig.add_trace(go.Scatter(x=dfns['time'], y=dfns['env_car']/1000, mode='lines', name='Car Sent', line=dict(color='blue', dash='dot')), row=1, col=3, secondary_y=False)
fig.add_trace(go.Scatter(x=dfns['time'], y=dfns['rec_serv']/1000, mode='lines', name='Car Received', line=dict(color='red', dash='dash')), row=1, col=3, secondary_y=False)

fig.add_trace(go.Scatter(x=dfss['time'], y=dfss['pdr'], mode='lines', name='PDR FS', line=dict(color='green', dash='dot', width=3)), row=2, col=1, secondary_y=False)
fig.add_trace(go.Scatter(x=dfqs['time'], y=dfqs['pdr'], mode='lines', name='PDR FQ', line=dict(color='green', dash='dot', width=3)), row=2, col=2, secondary_y=False)
fig.add_trace(go.Scatter(x=dfns['time'], y=dfns['pdr'], mode='lines', name='PDR FN', line=dict(color='green', dash='dot', width=3)), row=2, col=3, secondary_y=False)

# Atualizando layout para um design coerente
fig.update_layout(
    title='Comparative Performance Analysis - App S (Safety/MEC - 500Kbps - Prioridade 0)',
    xaxis_title='Time (s)',
    legend_title='Metrics',
    template='plotly_dark',
    height=700,  # Ajustar a altura se necessário
    #width=3700   # Ajustar a largura para caber os três plots
)

# Atualizando configurações dos eixos y
for i in range(1, 4):
    fig.update_yaxes(title_text="Throughput (Kbps)", range=[0, 700], row=1, col=i, secondary_y=False)
    fig.update_yaxes(title_text="Packet Delivery Rate (%)", row=2, col=i, secondary_y=False)
    fig.update_yaxes(title_text="RTT (ms)", range=[0, 40], row=1, col=i, secondary_y=True)  # Ajuste do range do eixo Y secundário

# Mostrando gráfico
fig.show()


# #### Define requisitos, classifica e analise impacto - App S (Safety/MEC - 500Kbps - Prioridade 0)

# In[17]:


# Requisitos de impacto App S (Safety/MEC - 500Kbps - Prioridade 0)
low = ['<=10', '>=85', '>350']
medium = ['<=20', '>=70', '>=250']
high = ['>20', '<70', '<250']
imp_s = {'low': low, 'medium':medium, 'high':high}
imp_s = pd.DataFrame.from_dict(imp_s)
imp_s.index = ['RTT (ms)', 'PDR (%)', 'Rx (Kbps)']
print('\nRequisitos de impacto App S (Safety/MEC - 500Kbps - Prioridade 0)\n')
imp_s = pd.DataFrame.from_dict(imp_s)
display(imp_s)

# Classificando S
df_sim_ohe.loc[(df_sim_ohe['ohe__app_S'] == 1) & (df_sim_ohe[rtt] <= 10) & (df_sim_ohe['pdr'] >= 0.85) & (df_sim_ohe['rec_serv'] >= 350000),'impact_level'] = 'low'
df_sim_ohe.loc[(df_sim_ohe['ohe__app_S'] == 1) & (df_sim_ohe[rtt] <= 20) & (df_sim_ohe['pdr'] >= 0.70) & (df_sim_ohe['rec_serv'] >= 250000) & (df_sim_ohe['impact_level'] != 'low'), 'impact_level'] = 'medium'
df_sim_ohe.loc[(df_sim_ohe['ohe__app_S'] == 1) & ((df_sim_ohe[rtt] > 20) | (df_sim_ohe['pdr'] < 0.70) | (df_sim_ohe['rec_serv'] < 250000)) & (df_sim_ohe['impact_level'] != 'medium'), 'impact_level'] = 'high'

# App S - analise impacto
imp_s_fs = pd.DataFrame(df_sim_ohe.query('`ohe__app_S` == 1 and `ohe__approach_FS` == 1').groupby(['impact_level']).count()['ohe__approach_FS'])
imp_s_fq = pd.DataFrame(df_sim_ohe.query('`ohe__app_S` == 1 and `ohe__approach_FQ` == 1').groupby(['impact_level']).count()['ohe__approach_FQ'])
imp_s_fn = pd.DataFrame(df_sim_ohe.query('`ohe__app_S` == 1 and `ohe__approach_FN` == 1').groupby(['impact_level']).count()['ohe__approach_FN'])
analise_s = imp_s_fs.join(imp_s_fq).join(imp_s_fn)
analise_s.columns = ['FS', 'FQ', 'FN']
print('\nAnalise de impacto App S (Safety/MEC - 500Kbps - Prioridade 0)\n')
display(analise_s.T.style.background_gradient())


# ### Consolidado - analise de impacto

# In[18]:


# Consolidado classes
imp_e.join(imp_e2, lsuffix='_e', rsuffix='_e2').join(imp_g).join(imp_s, lsuffix='_g', rsuffix='_s').T.rename(columns={'RTT(ms))': "RTT'(ms)"})


# In[19]:


# Consolidado impacto
analise_e.join(analise_e2, lsuffix='_e', rsuffix='_e2').join(analise_g).join(analise_s, lsuffix='_g', rsuffix='_s').T.sort_values(by=['low'], ascending=False).style.background_gradient()


# ### Pre-processamento

# In[20]:


# 1) cópia e mapeamento seguro do target
df_sim_ohe_copy = df_sim_ohe.drop(columns=['time']).copy()
map_dict = {'low': 0, 'medium': 1, 'high': 2}
df_sim_ohe_copy['impact_level'] = df_sim_ohe_copy['impact_level'].map(map_dict).astype('int8')

# 2) converte qualquer coluna 'object' em numérica (one-hot deve virar int8/uint8)
obj_cols = df_sim_ohe_copy.select_dtypes(include='object').columns
df_sim_ohe_copy[obj_cols] = df_sim_ohe_copy[obj_cols].apply(pd.to_numeric, errors='coerce').astype('float32')

# 3) também converte inteiros “nullable” para inteiros normais (se quiser)
nullable_ints = df_sim_ohe_copy.select_dtypes(include='Int8').columns
df_sim_ohe_copy[nullable_ints] = df_sim_ohe_copy[nullable_ints].astype('int8')

# 4)features e labels
features = df_sim_ohe_copy.drop(columns=['impact_level'])
labels = df_sim_ohe_copy['impact_level']

# 5) salvando
df_sim_ohe_copy.to_csv('raw_full.csv', index=False)
features.to_csv('features.csv', index=False) 
labels.to_csv('labels.csv', index=False) 


# In[1]:


#leitura
import pandas as pd
import time
df_sim_ohe_copy = pd.read_csv(r"C:\Users\sarai\Documents\doutorado\artigos\producao\vtc2024\raw_full.csv")
features = pd.read_csv(r"C:\Users\sarai\Documents\doutorado\artigos\producao\vtc2024\features.csv")
labels = pd.read_csv(r"C:\Users\sarai\Documents\doutorado\artigos\producao\vtc2024\labels.csv")['impact_level']


# In[22]:


len(features.columns)


# In[23]:


features.columns


# ### Analise preliminar dos dados

# In[24]:


df_sim_ohe_copy.columns


# In[25]:


df_sim_ohe_copy.info()


# In[26]:


df_sim_ohe_copy.describe()[['rec_serv',  'bc_rtt',  'rtt_change', 'pdr_change', 'rtt_mam', 'log_pdr', 'log2_rtt', 'log2_pdr', 'rtt_sqrd', 'pdr_sqrd', 'env_car', 'pdr', 'pdr_mam', 'rtt_masd', 'rtt']]


# In[27]:


df_sim_ohe_copy.describe()[['ohe__app_E',  'ohe__app_E2', 'ohe__app_G', 'ohe__app_S', 'ohe__approach_FN', 'ohe__approach_FQ', 'ohe__approach_FS', 'ohe__cat_req_hbmlmp', 'ohe__cat_req_lblllp', 'ohe__cat_req_mbclcp', 'ohe__cat_req_mbmlhp']]


# In[28]:


df_sim_ohe_copy.describe()


# In[29]:


#verificando correlações
corr = df_sim_ohe_copy.corr(method='pearson')          # agora inclui todas as colunas numéricas

# heat-map
plt.figure(figsize=(20, 15))
sns.heatmap(corr,
            cmap='coolwarm',
            vmax=1.0, vmin=-1.0,
            linewidths=.6,
            annot=False)                 # deixe True se quiser números
plt.title('Correlation matrix (after dtype fix)')
plt.show()


# In[30]:


# Analise features pré-processadas
corr_matrix = df_sim_ohe_copy.corr()

# Configurando o tamanho da figura
plt.figure(figsize=(20, 15))  # Ajuste as dimensões conforme necessário

sns.heatmap(corr_matrix, annot=True, linewidths = .6, cmap='coolwarm')
plt.show()


# In[31]:


corr_matrix.to_csv('corr_matrix.csv', index=False) 

subset = df_sim_ohe_copy[["pdr", "pdr_sqrd", "pdr_mam",
                          "log_pdr", "log2_pdr"]]

corr_mat = subset.corr(method="pearson").round(3)
print(corr_mat)


# In[32]:


corr_target = corr['impact_level'].sort_values(key=np.abs, ascending=False)
print(corr_target.head(15))


# In[33]:


from sklearn.feature_selection import SelectKBest, f_classif
kbest = SelectKBest(f_classif, k='all').fit(features, labels)
scores = (
    pd.Series(kbest.scores_, index=features.columns)
      .sort_values(ascending=False)
)
print(scores.head(15))          # top‑15
print(scores.tail(10))          # piores


# In[34]:


features.columns


# In[35]:


corr


# In[36]:


corr.shape


# ## Treinamento Deep Learning

# In[11]:


import numpy as np, pandas as pd, tensorflow as tf

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, f_classif

from scikeras.wrappers import KerasClassifier
from tensorflow.keras import regularizers

from sklearn.model_selection import ParameterGrid


# In[40]:


# ──────────────────────────────────────────────────────────────
# 1. Filtro de correlação (> threshold)  ───────────────────────
# ──────────────────────────────────────────────────────────────
class CorrFilter(BaseEstimator, TransformerMixin):
    """Remove colunas com |ρ| >= threshold (default 0.90)."""
    def __init__(self, threshold=0.90):
        self.threshold = threshold
        self.to_drop_  = None
        
    def fit(self, X, y=None):
        # matriz de correlação absoluta (apenas colunas numéricas)
        corr = pd.DataFrame(X).corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        self.to_drop_ = [col for col in upper.columns
                         if any(upper[col] >= self.threshold)]
        return self
    
    def transform(self, X):
        return pd.DataFrame(X).drop(columns=self.to_drop_, errors="ignore").values
    
    # para o SelectKBest saber o nº de features pós‑filtro
    def get_feature_names_out(self, names=None):
        if names is None:
            return None
        return [n for n in names if n not in self.to_drop_]

class CorrFilterBest(BaseEstimator, TransformerMixin):
    """
    Para cada grupo de colunas com |ρ| ≥ threshold,
    mantém a de maior F‑score ANOVA calculado no treino.
    """
    def __init__(self, threshold=0.90):
        self.threshold = threshold
        self.to_drop_  = None          # será preenchido no fit()

    def fit(self, X, y):
        # ------- 1. F‑scores no subset de treino -------
        F, _ = f_classif(X, y)                       # retorna array
        F_scores = pd.Series(F, index=X.columns)

        # ------- 2. matriz de correlação --------------
        corr  = X.corr().abs()
        mask  = np.triu(np.ones(corr.shape), k=1).astype(bool)
        upper = corr.where(mask)

        # ------- 3. decidir quem remover -------------
        drop = []
        for col in upper.columns:
            twins = upper.index[upper[col] >= self.threshold].tolist()
            if twins:
                group = [col] + twins          # col + todos correlacionados ≥thr
                keep  = F_scores[group].idxmax()   # maior F dentro do grupo
                drop.extend([c for c in group if c != keep])
        self.to_drop_ = list(set(drop))
        return self

    def transform(self, X):
        return X.drop(columns=self.to_drop_, errors="ignore")


# In[47]:


class F1Macro(tf.keras.metrics.Metric):
    """F1-macro como métrica Keras."""
    
    def __init__(self, num_classes=3, name="f1_macro", dtype=tf.float32, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.num_classes = num_classes
        # acumuladores por classe
        self.tp = self.add_weight(name="tp", shape=(num_classes,), initializer="zeros", dtype=dtype)
        self.fp = self.add_weight(name="fp", shape=(num_classes,), initializer="zeros", dtype=dtype)
        self.fn = self.add_weight(name="fn", shape=(num_classes,), initializer="zeros", dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true esperado como ints; remove 2ª dimensão caso venha (batch,1)
        y_true = tf.cast(y_true, tf.int32)
        
        # Use tf.shape para trabalhar com dynamic shapes
        y_true_shape = tf.shape(y_true)
        
        # Remove a 2ª dimensão SE rank > 1
        if len(y_true.shape.as_list()) > 1:
            y_true = tf.squeeze(y_true, axis=-1)
        
        # classe prevista = argmax das probabilidades
        y_pred = tf.argmax(y_pred, axis=1, output_type=tf.int32)
        
        # one-hot para cálculo vetorizado
        y_true_oh = tf.one_hot(y_true, depth=self.num_classes, dtype=self.dtype)
        y_pred_oh = tf.one_hot(y_pred, depth=self.num_classes, dtype=self.dtype)
        
        if sample_weight is not None:
            # Garante o tipo correto primeiro
            sample_weight = tf.cast(sample_weight, self.dtype)
            
            # Obter o shape dinâmico de sample_weight
            sw_shape = tf.shape(sample_weight)
            
            # Reshape para garantir shape (batch_size, 1)
            sample_weight = tf.reshape(sample_weight, [-1, 1])
            
            # Difunde (broadcast) os pesos para os vetores one-hot
            y_true_oh *= sample_weight
            y_pred_oh *= sample_weight
        
        # contagens em lote
        tp_batch = tf.reduce_sum(y_true_oh * y_pred_oh, axis=0)
        fp_batch = tf.reduce_sum((1. - y_true_oh) * y_pred_oh, axis=0)
        fn_batch = tf.reduce_sum(y_true_oh * (1. - y_pred_oh), axis=0)
        
        # acumula
        self.tp.assign_add(tp_batch)
        self.fp.assign_add(fp_batch)
        self.fn.assign_add(fn_batch)

    def result(self):
        precision = tf.math.divide_no_nan(self.tp, self.tp + self.fp)
        recall = tf.math.divide_no_nan(self.tp, self.tp + self.fn)
        f1 = tf.math.divide_no_nan(2. * precision * recall, precision + recall)
        return tf.reduce_mean(f1)  # macro-average

    def reset_state(self):
        for var in (self.tp, self.fp, self.fn):
            var.assign(tf.zeros_like(var))


# In[48]:


from keras.saving import register_keras_serializable

@register_keras_serializable()
class F1Macro(tf.keras.metrics.Metric):
    """F1-macro como métrica Keras."""
    
    def __init__(self, num_classes=3, name="f1_macro", dtype=tf.float32, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.num_classes = num_classes
        # acumuladores por classe
        self.tp = self.add_weight(name="tp", shape=(num_classes,), initializer="zeros", dtype=dtype)
        self.fp = self.add_weight(name="fp", shape=(num_classes,), initializer="zeros", dtype=dtype)
        self.fn = self.add_weight(name="fn", shape=(num_classes,), initializer="zeros", dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true esperado como ints; remove 2ª dimensão caso venha (batch,1)
        y_true = tf.cast(y_true, tf.int32)
        
        # Use tf.shape para trabalhar com dynamic shapes
        y_true_shape = tf.shape(y_true)
        
        # Remove a 2ª dimensão SE rank > 1
        if len(y_true.shape.as_list()) > 1:
            y_true = tf.squeeze(y_true, axis=-1)
        
        # classe prevista = argmax das probabilidades
        y_pred = tf.argmax(y_pred, axis=1, output_type=tf.int32)
        
        # one-hot para cálculo vetorizado
        y_true_oh = tf.one_hot(y_true, depth=self.num_classes, dtype=self.dtype)
        y_pred_oh = tf.one_hot(y_pred, depth=self.num_classes, dtype=self.dtype)
        
        if sample_weight is not None:
            # Garante o tipo correto primeiro
            sample_weight = tf.cast(sample_weight, self.dtype)
            
            # Obter o shape dinâmico de sample_weight
            sw_shape = tf.shape(sample_weight)
            
            # Reshape para garantir shape (batch_size, 1)
            sample_weight = tf.reshape(sample_weight, [-1, 1])
            
            # Difunde (broadcast) os pesos para os vetores one-hot
            y_true_oh *= sample_weight
            y_pred_oh *= sample_weight
        
        # contagens em lote
        tp_batch = tf.reduce_sum(y_true_oh * y_pred_oh, axis=0)
        fp_batch = tf.reduce_sum((1. - y_true_oh) * y_pred_oh, axis=0)
        fn_batch = tf.reduce_sum(y_true_oh * (1. - y_pred_oh), axis=0)
        
        # acumula
        self.tp.assign_add(tp_batch)
        self.fp.assign_add(fp_batch)
        self.fn.assign_add(fn_batch)

    def result(self):
        precision = tf.math.divide_no_nan(self.tp, self.tp + self.fp)
        recall = tf.math.divide_no_nan(self.tp, self.tp + self.fn)
        f1 = tf.math.divide_no_nan(2. * precision * recall, precision + recall)
        return tf.reduce_mean(f1)  # macro-average

    def reset_state(self):
        for var in (self.tp, self.fp, self.fn):
            var.assign(tf.zeros_like(var))
    
    # Métodos para serialização
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

from keras.utils import get_custom_objects
get_custom_objects()["F1Macro"] = F1Macro


# In[49]:


# Minimal Model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(10,)),
    tf.keras.layers.Dense(5, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax') # Assuming 3 classes
])

# Try compiling with the metric
try:
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=[F1Macro(num_classes=3)]) # Instantiate it
    print("Compilation successful!")

    # Optional: Try a dummy fit
    # print("Trying dummy fit...")
    # dummy_x = np.random.rand(16, 10)
    # dummy_y = np.random.randint(0, 3, size=(16,))
    # model.fit(dummy_x, dummy_y, epochs=1, verbose=0)
    # print("Dummy fit successful!")

except Exception as e:
    print(f"Error during isolated test: {e}")
    import traceback
    traceback.print_exc()


# ### Deep Network basic DNN (MLP)

# #### Encontra parametros ótimos (MLP)

# In[50]:


# ------------------------------------------------------------
# 1) Função construtora do MLP
# ------------------------------------------------------------
# • `meta`   → dicionário que o SciKeras injeta com infos do fold
#              (inclui o nº de atributos já pós‑pipeline).
# • `l2_val` → fator de regularização L2 (weight‑decay).
# • `drop`   → taxa de Dropout aplicada após cada camada densa.
# • `n1/n2`  → nº de neurônios nas camadas ocultas 1 e 2.
# O modelo compila em modo “sparse” (labels inteiros 0‑2) e
# devolve rede pronta para ser embrulhada pelo KerasClassifier.

def build_mlp(meta, l2_val, drop, n1, n2):
    tf.keras.backend.clear_session()                 # limpa grafos antigos
    n_inputs = meta["n_features_in_"]          # nº de colunas que entram
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_inputs,)),                             # camada de entrada
        tf.keras.layers.Dense(n1, activation="relu",                          # hidden‑1
                              kernel_regularizer=regularizers.l2(l2_val)),
        tf.keras.layers.Dropout(drop),
        tf.keras.layers.Dense(n2, activation="relu",                          # hidden‑2
                              kernel_regularizer=regularizers.l2(l2_val)),
        tf.keras.layers.Dropout(drop),
        tf.keras.layers.Dense(3, activation="softmax")                        # saída 3 classes
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy", F1Macro(num_classes=3)])
    
    # força a criação de um grafo concreto apenas UMA vez
    model.train_function = model.make_train_function()
    model.test_function  = model.make_test_function()
    return model

# ------------------------------------------------------------
# 2) Wrapper SciKeras  → permite tunar hiper‑parâmetros Keras
# ------------------------------------------------------------

# callbacks -------------------------------------------------
early_stop = tf.keras.callbacks.EarlyStopping(
               patience=10,                # Treinamento para se val_loss não baixar por 10 épocas consecutivas
               restore_best_weights=True, # Após parar, volta aos pesos da época com menor val_loss (evita “over‑treinar”).
               #monitor="val_accuracy", mode="max",  # já é calculado - métrica observada é a acurácia (os 20 % definidos por validation_split=0.20)
               #monitor="val_loss", mode="min",
               monitor="val_f1_macro", mode="max", # quanto menor a acuracia, melhor
               verbose = 1) 

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
               monitor="val_loss",     # ou "val_accuracy" / "val_f1"
               mode="min",             # "min" se métrica = loss, "max" se accuracy/F1
               factor=0.3,             # novo_LR = LR_atual × 0.3
               patience=4,             # nº de épocas sem melhora
               min_lr=1e-6,            # piso
               verbose=1)

keras_clf = KerasClassifier(
        model             = build_mlp,            # função acima
        epochs            = 200,                  # máx. de épocas
        batch_size        = 16,
        verbose           = 1,
        validation_split  = 0.20,                 # 20 % do treino p/ early‑stop
        callbacks=[early_stop, reduce_lr]       
)

# ------------------------------------------------------------
# 3) Pipeline  (Scaler ➜  Rede)  ***sem leakage***
#    – qualquer transformação extra (SelectKBest, CorrFilter)
#      pode ser inserida ANTES do scaler.
# ------------------------------------------------------------
pipe = Pipeline([
        ("kbest" , SelectKBest(f_classif, k=20)),
        ("scale", StandardScaler()),
        ("clf"  , keras_clf)
])

# ------------------------------------------------------------
# 4) Espaço de busca para RandomizedSearchCV
#    prefixo "clf__model__" =  parámetro do modelo dentro do wrapper
# ------------------------------------------------------------

param_dist = {
    # ........................ hiper‑parâmetros do modelo ......................
    "clf__model__l2_val": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    "clf__model__drop"  : [0.25, 0.35, 0.45, 0.55],
    "clf__model__n1"    : [16, 32, 48],
    "clf__model__n2"    : [8, 16, 24],

    # ........................ hiper‑parâmetro do SelectKBest .................
    "kbest__k": [10, 15, 20, 25]          #  ←  aqui!
}

# 5‑fold estratificado (mesmo split usado em toda a otimização) - mantém a proporção
#Validar também com 10 folds
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ------------------------------------------------------------
# 5) Busca aleatória (20 configurações × 5 folds = 100 treinamentos)
#    • scoring = f1_macro  → balanceia classes
#    • n_jobs = 1          → evita problemas de pickling com Keras
# ------------------------------------------------------------

total = len(ParameterGrid(param_dist))

#with parallel_backend("threading"):            # usa todos os núcleos sem pickling
#GridSearch - 
rand = RandomizedSearchCV(
    estimator            = pipe,
    param_distributions  = param_dist,
    n_iter               = min(total, 20),
    cv                   = cv,
    scoring              = "f1_macro",
    n_jobs               = -1,
    refit                = True,            # refaz fit no melhor conjunto
    verbose              = 0,
    random_state         = 42)
# --------------------------------------------------------
# 6) Execução
# --------------------------------------------------------
t0 = time.perf_counter()                 # início
rand.fit(features, labels)

# --------------------------------------------------------
# 7) Resultados
# --------------------------------------------------------
print("Melhor F1‑macro: ", rand.best_score_)
print("Hiper‑parâmetros:", rand.best_params_)

t_total = time.perf_counter() - t0       # ⏱️  fim
print(f"\nTempo total: {t_total/60:.1f} min ({t_total:.1f} s)")

cv_times = pd.DataFrame({
        "fit (s)"  : rand.cv_results_["mean_fit_time"],
        "score (s)": rand.cv_results_["mean_score_time"]
})
print(cv_times.describe())        # média, std, etc.


# #### Executa com parâmetros ótimos (MLP)

# In[51]:


# pipeline já vencedor; não há mais RandomizedSearch aqui
best_final = rand.best_estimator_

cv = StratifiedKFold(5, shuffle=True, random_state=42)
scoring = {"accuracy": "accuracy",
           "precision_macro": "precision_macro",
           "recall_macro":    "recall_macro",
           "f1_macro":        "f1_macro"}

cv_res = cross_validate(best_final, features, labels,
                        cv=cv, scoring=scoring, n_jobs=1, return_train_score=False)

for m in scoring:
    vals = cv_res[f"test_{m}"]
    print(f"{m:15s}: mean = {vals.mean():.4f} | std = {vals.std():.4f}")

for m, v in scoring.items():
    print(f"{m:16s}: {cv_res[f'test_{m}'].mean():.4f}")


# In[18]:


for m, v in scoring.items():
    print(f"{m:16s}: {cv_res[f'test_{m}'].mean():.4f}")


# In[6]:


rand.fit(features, labels)

print("Melhor f1_macro :", rand.best_score_)
print("Melhores params :", rand.best_params_)


# #### Executa com parâmetros ótimos (MLP)

# In[7]:


#construtora modelo
def build_best(meta, n1=96, n2=32, l2_val=1e-05, drop=0.1):
    n_inputs = meta["n_features_in_"]
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_inputs,)),
        tf.keras.layers.Dense(n1, activation="relu", kernel_regularizer=regularizers.l2(l2_val)),
        tf.keras.layers.Dropout(drop),
        tf.keras.layers.Dense(n2, activation="relu", kernel_regularizer=regularizers.l2(l2_val)),
        tf.keras.layers.Dropout(drop),
        tf.keras.layers.Dense(3, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# In[8]:


#Parametros de treino
best_clf = KerasClassifier(
        model           = build_best,
        epochs          = 100,
        batch_size      = 32,
        verbose         = 0,
        validation_split= 0.2,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss")]
)

#pipeline para normalizar e depois modelar
final_pipe = Pipeline([
        ("scale", StandardScaler()),
        #("kbest" , SelectKBest(f_classif, k=25)),
        ("clf"  , best_clf)
])

# ---------- 3. Cross‑validation ----------
cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {"accuracy"       : "accuracy",
           "precision_macro": "precision_macro",
           "recall_macro"   : "recall_macro",
           "f1_macro"       : "f1_macro"}


# In[9]:


#Executa treino/teste com cross-validation
cv_res_mlp = cross_validate(final_pipe, features, labels, cv=cv, scoring=scoring, n_jobs=1, return_train_score=False)

#Imprime resultados
for m in scoring:
    vals = cv_res_mlp[f"test_{m}"]
    print(f"{m:16s}:  mean = {vals.mean():.4f} | std = {vals.std():.4f}")


# In[10]:


from joblib import parallel_backend
with parallel_backend("threading"):
    rand = RandomizedSearchCV(pipe,
                              param_distributions=param_dist,
                              n_iter=20,
                              cv=cv,
                              scoring="f1_macro",
                              n_jobs=-1,   # agora usa todos os cores
                              verbose=1,
                              random_state=42)
    rand.fit(features, labels)


# In[12]:


print("Melhor f1_macro :", rand.best_score_)
print("Melhores params :", rand.best_params_)


# ### LSTM

# In[ ]:


# =============================================================
# 5‑fold CV   |   StandardScaler ➜ SelectKBest(k=30) ➜ LSTM
# =============================================================

# ---------- 1. build_fn LSTM (usa meta) ----------
def build_lstm(meta, n_classes=3):
    n_inputs = meta["n_features_in_"]               # ← nº de colunas após SelectKBest
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_inputs,)),   # 2‑D
        tf.keras.layers.Reshape((1, n_inputs)),     # 3‑D  [time‑steps=1, feats]
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

clf = KerasClassifier(
        model            = build_lstm,          # input_dim é dado por meta
        epochs           = 100,
        batch_size       = 32,
        verbose          = 0,
        validation_split = 0.20,
        callbacks=[tf.keras.callbacks.EarlyStopping(
                     patience=5, restore_best_weights=True, monitor="val_loss")]
)

# ---------- 2. Pipeline leakage‑free ----------
pipe = Pipeline([
    ("corr"  , CorrFilterBest(threshold=0.90)),          # remove colineares
    ("scale" , StandardScaler()),
    ("kbest" , SelectKBest(f_classif, k=15)),        # escolhe top‑15 restantes
    ("clf"   , clf)
])

pipe2 = Pipeline([
    ("scale" , StandardScaler()),
    #("corr"  , CorrFilter(threshold=0.90)),          # remove colineares
    ("kbest" , SelectKBest(f_classif, k=15)),        # escolhe top‑15 restantes
    ("clf"   , clf)
])

# ---------- 3. Cross‑validation ----------
cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {"accuracy"       : "accuracy",
           "precision_macro": "precision_macro",
           "recall_macro"   : "recall_macro",
           "f1_macro"       : "f1_macro"}

cv_res_lstm = cross_validate(pipe, features, labels,
                        cv=cv, scoring=scoring, n_jobs=1,
                        return_train_score=False)

for m in scoring:
    vals = cv_res_lstm[f"test_{m}"]
    print(f"{m:16s}:  mean = {vals.mean():.4f} | std = {vals.std():.4f}")


cv_res_lstm2 = cross_validate(pipe2, features, labels,
                        cv=cv, scoring=scoring, n_jobs=1,
                        return_train_score=False)
for m in scoring:
    vals = cv_res_lstm2[f"test_{m}"]
    print(f"{m:16s}:  mean = {vals.mean():.4f} | std = {vals.std():.4f}")


# ### RNN

# In[ ]:


# =============================================================
# 5‑fold CV   |   StandardScaler ➜ SelectKBest(k=30) ➜ SimpleRNN
# =============================================================
from sklearn.model_selection  import StratifiedKFold, cross_validate
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing    import StandardScaler
from sklearn.pipeline         import Pipeline
from scikeras.wrappers        import KerasClassifier
import tensorflow as tf

# ---------- 1. build_fn para SimpleRNN ----------
def build_rnn(meta, n_classes=3):
    n_inputs = meta["n_features_in_"]            # nº de colunas após SelectKBest
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_inputs,)),
        tf.keras.layers.Reshape((1, n_inputs)),  # → 3‑D [steps=1, feats]
        tf.keras.layers.SimpleRNN(50),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

clf = KerasClassifier(
        model            = build_rnn,     # input_dim vem do meta
        epochs           = 100,
        batch_size       = 32,
        verbose          = 0,
        validation_split = 0.20,
        callbacks=[tf.keras.callbacks.EarlyStopping(
                     patience=5, restore_best_weights=True, monitor="val_loss")]
)

# ---------- 2. Pipeline leakage‑free ----------
pipe = Pipeline([
    ("corr"  , CorrFilterBest(threshold=0.90)),          # remove colineares
    ("scale" , StandardScaler()),
    ("kbest" , SelectKBest(f_classif, k=15)),        # escolhe top‑15 restantes
    ("clf"   , clf)
])

pipe2 = Pipeline([
    ("scale" , StandardScaler()),
    #("corr"  , CorrFilter(threshold=0.90)),          # remove colineares
    ("kbest" , SelectKBest(f_classif, k=15)),        # escolhe top‑15 restantes
    ("clf"   , clf)
])

# ---------- 3. Cross‑validation ----------
cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {"accuracy"       : "accuracy",
           "precision_macro": "precision_macro",
           "recall_macro"   : "recall_macro",
           "f1_macro"       : "f1_macro"}

cv_res_rnn = cross_validate(pipe, features, labels,
                        cv=cv, scoring=scoring, n_jobs=1,
                        return_train_score=False)

for m in scoring:
    vals = cv_res_rnn[f"test_{m}"]
    print(f"{m:16s}:  mean = {vals.mean():.4f} | std = {vals.std():.4f}")


cv_res_rnn2 = cross_validate(pipe2, features, labels,
                        cv=cv, scoring=scoring, n_jobs=1,
                        return_train_score=False)
for m in scoring:
    vals = cv_res_rnn2[f"test_{m}"]
    print(f"{m:16s}:  mean = {vals.mean():.4f} | std = {vals.std():.4f}")


# ### GRU

# In[ ]:


# =============================================================
# 5‑fold CV   |   StandardScaler ➜ SelectKBest(k=30) ➜ GRU
# =============================================================
from sklearn.model_selection  import StratifiedKFold, cross_validate
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing    import StandardScaler
from sklearn.pipeline         import Pipeline
from scikeras.wrappers        import KerasClassifier
import tensorflow as tf

# ---------- 1. build_fn GRU ----------
def build_gru(meta, n_classes=3):
    n_inputs = meta["n_features_in_"]            # colunas após SelectKBest
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_inputs,)),
        tf.keras.layers.Reshape((1, n_inputs)),  # 3‑D  [steps=1, feats]
        tf.keras.layers.GRU(50),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

clf = KerasClassifier(
        model            = build_gru,
        epochs           = 100,
        batch_size       = 32,
        verbose          = 0,
        validation_split = 0.20,
        callbacks=[tf.keras.callbacks.EarlyStopping(
                     patience=5, restore_best_weights=True, monitor="val_loss")]
)

# ---------- 2. Pipeline leakage‑free ----------
pipe = Pipeline([
    ("corr"  , CorrFilterBest(threshold=0.90)),          # remove colineares
    ("scale" , StandardScaler()),
    ("kbest" , SelectKBest(f_classif, k=15)),        # escolhe top‑15 restantes
    ("clf"   , clf)
])

pipe2 = Pipeline([
    ("scale" , StandardScaler()),
    #("corr"  , CorrFilter(threshold=0.90)),          # remove colineares
    ("kbest" , SelectKBest(f_classif, k=15)),        # escolhe top‑15 restantes
    ("clf"   , clf)
])

# ---------- 3. Cross‑validation ----------
cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {"accuracy"       : "accuracy",
           "precision_macro": "precision_macro",
           "recall_macro"   : "recall_macro",
           "f1_macro"       : "f1_macro"}

cv_res_gru = cross_validate(pipe, features, labels,
                        cv=cv, scoring=scoring, n_jobs=1,
                        return_train_score=False)

for m in scoring:
    vals = cv_res_gru[f"test_{m}"]
    print(f"{m:16s}:  mean = {vals.mean():.4f} | std = {vals.std():.4f}")


cv_res_gru2 = cross_validate(pipe2, features, labels,
                        cv=cv, scoring=scoring, n_jobs=1,
                        return_train_score=False)
for m in scoring:
    vals = cv_res_gru2[f"test_{m}"]
    print(f"{m:16s}:  mean = {vals.mean():.4f} | std = {vals.std():.4f}")


# In[ ]:


#Verificando disribuiçao para cross-validation (ex. k=5)
#print(y_train.value_counts(normalize=False))
#print(y_test.value_counts(normalize=False))


# In[ ]:


scoring


# In[ ]:


cv_res


# ## Gráficos de analise DL

# In[ ]:


#grafico consolidado

import matplotlib.pyplot as plt

# Resultados apurados nas avaliações
models = ['DNN', 'LSTM', 'RNN', 'GRU']
accuracy = [accuracy_mlp, accuracy_lstm, accuracy_rnn, accuracy_gru]
recall = [recall_mlp, recall_lstm, recall_rnn, recall_gru]
precision = [precision_mlp, precision_lstm, precision_rnn, precision_gru]
f1 = [f1_mlp, f1_lstm, f1_rnn, f1_gru]

# Criar figura e eixos para um gráfico
fig, ax = plt.subplots(2, 2, figsize=(15, 10))

# Ajustar o espaço entre os gráficos
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# Gráfico de barras para Acurácia
ax[0, 0].bar(models, accuracy, color='skyblue')
ax[0, 0].set_title('Accuracy (%)')
ax[0, 0].set_ylim(0.75, 1)  # Definir limites do eixo y para uniformidade
for i, v in enumerate(accuracy):
    ax[0, 0].text(i, v + 0.005, "{:.2%}".format(v), ha='center')

# Gráfico de barras para Recall
ax[0, 1].bar(models, recall, color='lightgreen')
ax[0, 1].set_title('Recall (%)')
ax[0, 1].set_ylim(0.75, 1)
for i, v in enumerate(recall):
    ax[0, 1].text(i, v + 0.005, "{:.2%}".format(v), ha='center')

# Gráfico de barras para Precision
ax[1, 0].bar(models, precision, color='lightcoral')
ax[1, 0].set_title('Precision (%)')
ax[1, 0].set_ylim(0.75, 1)
for i, v in enumerate(precision):
    ax[1, 0].text(i, v + 0.005, "{:.2%}".format(v), ha='center')

# Gráfico de barras para F1-Score
ax[1, 1].bar(models, f1, color='orchid')
ax[1, 1].set_title('F1-Score (%)')
ax[1, 1].set_ylim(0.75, 1)
for i, v in enumerate(f1):
    ax[1, 1].text(i, v + 0.005, "{:.2%}".format(v), ha='center')

#Salvando gráfico (imagem)
plt.savefig('consolidadov2.png')

# Mostrar o gráfico consolidado
plt.show()


# ## Impact distribution

# In[ ]:


# Convertendo os dados one-hot encoded para uma coluna 'Application'
application_mapping = {
    'ohe__app_E': 'E',
    'ohe__app_E2': 'E2',
    'ohe__app_G': 'G',
    'ohe__app_S': 'S'
}
df_sim_ohe['Application'] = df_sim_ohe[[*application_mapping.keys()]].idxmax(axis=1).map(application_mapping)

# Contando as frequências de impacto por aplicação
impact_distribution = df_sim_ohe.groupby(['Application', 'impact_level']).size().unstack(fill_value=0)

# Calculando porcentagens
impact_distribution_percentage = impact_distribution.div(impact_distribution.sum(axis=1), axis=0) * 100

# Criando o gráfico
impact_distribution_percentage.plot(kind='bar', stacked=True, colormap='viridis', figsize=(10, 6))
plt.title('Percentage Distribution of Impact Levels by Application', fontsize=14)
plt.xlabel('Application', fontsize=12)
plt.ylabel('Percentage (%)', fontsize=12)
plt.legend(title='Impact Level', fontsize=10)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# In[ ]:




