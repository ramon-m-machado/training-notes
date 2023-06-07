# SHAP
[reference paper](https://repositorium.sdum.uminho.pt/bitstream/1822/23526/1/nsensitivity2.pdf)

vem da teoria dos jogos
exemplo do pagamento proporcional a quanto cada membro da equipe ajudou
distribuição justa -> shapley


shap 
players -> features ( contribute different)

pode tirar uma feature e ver o qt isso interfere, mas tem que considerar a interação
considerar todos os diferentes subsets of players e fazer a media
-> contribuição marginal

-> local explanations
- agregando as funcoes locais é possivel ter explicações globais
![image](https://github.com/ramon-m-machado/training-notes/assets/86575893/418cfc76-44aa-4dfd-bb4c-619d0014a418)

-> como excluir features? botar valores random do dataset 
-> complexity 2**n


---> aproximate SHAP
Kernel SHAP
faz umas samples e um modelo de regressão linear baseados neles
onde os valores sao 1 e 0 pra feature presente ou nao
-> os coef podem ser interpretados como uma aproximação do shap values
(similar pro lime, mas aqui não usa o peso da distancia, e sim o peso da informação)

tem outros shaps






# LIME
[reference paper](https://arxiv.org/pdf/1602.04938.pdf)
Local Interpretable Model-agnostic Explanations (LIME)

O objetivo geral do LIME é identificar um modelo interpretável sobre a representação interpretável que seja localmente fiel ao classificador.


"The overall goal of LIME is to identify an interpretable model over the interpretable representation that is locally faithful to the classifier."

considera o modelo como caixa preta, nao faz suposições sobre como o modelo funciona
 
 tem um certo input que desejamos explicar e lime gera outros exemplos perto dele
 ( aqui explicar o que seria perto)
 ( e explicar pra imagem)
 
 cria um modelo linear local para explicar como o modelo se comporta na vizinhança
 
 
 
 como o algoritmo classifica globalmente pode ser dificil, porém localmente se aproxima de uma reta, é simples e linear
 
 random weighted (closer) samples around the input
  
  
  mostrar wolf vs husky
  
  
falando sobre lime
- muito citado
- relativamente simples de entender
- relativamente simples para implementar
- muito popular


-> dont endup with a blackbox explaining another blackbox


limitações:
- assume linearidade local
- caro se for gerar para todo o dataset (5000 samples para cada input)
- instabilidade já que o lime gera amostras aleatórias em torno do seu input, pode haver explicações diferentes se você rodar o LIME com o mesmo input duas vezes, como mostrado no paper de 2019
- vulnerável a ataques, teve um paper de 2020 que mostrou um modelo intencionalmente feito com viés, que não era detectado pelo LIME

. Criar pertubações da imagem

2. Predict the class of each artificial data point
3. Calculate the weight of each artificial data point
4. Fit a linear classifier to explain the most important features
os coeficientes da regressao vao te dar para cada dimensao (superpixel) um coeficiente que pode ser interpretado como a importancia dele
pega os n mais importantes e voce consegue plotar

Podemos, assim, recuperar os N superpixels mais importantes e mostrá-los:
