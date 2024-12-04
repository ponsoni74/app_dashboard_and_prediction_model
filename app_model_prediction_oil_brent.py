import streamlit as st
import pandas as pd
#import numpy as np
#from keras.models import Sequential
#from keras.layers import LSTM, Dense
import joblib
#import streamlit.components.v1 as components
#from sklearn.preprocessing import MinMaxScaler
##from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

st.title('Modelo de previsão do preço do petróleo Brent para o dia 18/11/2024')
st.write('Obs.: Informamos que em breve a data para previsão poderá ser selecionada')
         
# importando os dados
data = pd.read_excel('preco_petroleo_brent_12-05-2000.xlsx', parse_dates=['Data'], index_col='Data')

# Normalização dos dados
#scaler = MinMaxScaler(feature_range=(0,1))
#scaled_data = scaler.fit_transform(data)

# Preparação dos dados com a geração do array de entrada X (sequencias), e do array de saída equivalente y.
def create_sequences(data, sequence_length):
  X_sequences, y_targets = [], []

  for i in range(len(data) - sequence_length):
    X_sequences.append(data[i:i+sequence_length, 0])  #scaled_data
    y_targets.append(data[i+sequence_length, 0])  #scaled_data

  return np.array(X_sequences), np.array(y_targets)

sequence_length = 10
X, y = create_sequences(data, sequence_length)   #scaled_data

# Separação dos dados entre conjunto de dados para treinamento e conjunto de dados para teste.
train_size = int(len(X) * 0.8)
X_train, X_test = X[train_size:], X[:train_size]
y_train, y_test = y[train_size:], y[:train_size]

# Construindo o modelo LSTM
#model = Sequential()
#model.add(LSTM(units=64, input_shape=(sequence_length, 1)))
#model.add(Dense(units=1))
#model.compile(optimizer='adam', loss='mean_squared_error')
#model.summary()

# Treinando o modelo
#model.fit(X_train, y_train, epochs=10, batch_size=32)

# importando o modelo preditivo
model = joblib.load('model.pkl')

# Gerar previsão
predictions = model.predict(X_test)

#mse = mean_squared_error(y_test, predictions)
#rmse = np.sqrt(mse)
#mae = mean_absolute_error(y_test, predictions)
#mae = mean_absolute_error(y_test, predictions)

# Revertendo as escalas para plotar os valores na escala real do índice
#predicted_prices = scaler.inverse_transform(predictions)
#actual_prices = scaler.inverse_transform(y_test.reshape(-1,1))

##mape = mean_absolute_percentage_error(actual_prices, predicted_prices)

preco_atual = pd.DataFrame(data)
preco_potencial = pd.DataFrame(data)

st.write(f'O preço real é ........................... US$  {preco_atual.iloc[0][0].round(2)}')
st.write(f'A previsão do preço é................ US$ {preco_potencial.iloc[0][0].round(2)}')

erro = abs(((preco_potencial.iloc[0][0].round(4) - preco_atual.iloc[0][0].round(2))/ preco_atual.iloc[0][0].round(2)*100).round(2))

st.write(f'Erro na previsão em % foi de.................. {erro.round(2)}%')

#st.write(f'Mean Squared Error (MSE)...................... {mse:.4f}')
#st.write(f'Root Mean Squared Error (RMSE)................ {rmse:.4f}')
#st.write(f'Mean Absolute Error (MAE)......................... {mae:.4f}')
#st.write(f'Mean Absolute Percentage Error (MAPE)...... {mape:.4f}')

#criar o gráfico
st.write('Gráfico com a evolução de preço real do barril')
st.line_chart(data)

# filtro para o gráfico
#st.sidebar.markdown('### Filtro para o gráfico')

# Exibir resultado
#st.write("Previsão do preço em US$ para os próximos {} dias:".format(days))
#st.write(forecast[['ds', 'yhat']].tail(days))

#categoria_grafico = st.sidebar.selectbox('Selecione a categoria para apresentar no gráfico', options = dados['Categoria'].unique())
#figura = plot_estoque(dados, categoria_grafico)
#st.pyplot(figura)
