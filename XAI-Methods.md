# SHAP
[reference paper](https://repositorium.sdum.uminho.pt/bitstream/1822/23526/1/nsensitivity2.pdf)
* f(x) prediction based on x
* x singular input
* x' simplified input
* hx(x') = x  in wich h is a mapping function, especific for input x
* g(z') ~= f(hx(z'))  whenever z' ~= x'





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

