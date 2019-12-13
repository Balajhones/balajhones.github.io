---
layout: post
title:  "Visão computacional para indústrias"
date:   2019-12-11 15:39:56 -0300
categories: jekyll update
---

As indústrias possuem meios ainda manuais para inspeção de peças, a fim de garantir que o produto fabricado seja 
funcional e tenha garantia de qualidade. Para isso são necessários funcionários específicos para esta atividade, 
que analisam peça por peça, conforme legislação e qualificação, para encontrar defeitos no produto. Entretanto, 
é um trabalho extremamente cansativo e passível a muitas falhas. 
O objetivo deste artigo é trazer uma solução para este serviço manual e desgastante, onde substituiriam as etapas 
de análise e verificação das inspeções das peças por um modelo  automatizado que identificaria se a peça através de 
imagens, possui anormalidades. Baseado em estudos de Deep Learning de imagem, através de treino e teste terá a 
capacidade de distinguir cada peça, o que envolve conhecimentos de Matemática, Estatística, Programação, 
Visão Computacional, Pré-Processamento de imagens, entre outras áreas.
Introdução – Redes neurais

A arquitetura da rede neural refere-se a elementos como o número de camadas na rede, o número de unidades em cada 
camada e como as unidades são conectadas entre as camadas. As redes neurais são vagamente inspiradas no 
funcionamento do cérebro humano. Assim como os neurônios transmitem sinais pelo cérebro, as unidades tomam alguns 
valores das unidades anteriores como entrada, realizam uma computação e, em seguida, transmitem o novo valor como 
saída para outras unidades. Essas unidades são colocadas em camadas para formar a rede, iniciando no mínimo com uma 
camada para entrada de valores e uma camada para valores de saída. O termo hidden layer ou camada oculta é usado para
 todas as camadas entre as camadas de entrada e saída, ou seja, aquelas “ocultas” do mundo real.
Arquiteturas diferentes podem produzir resultados drasticamente diferentes, já que o desempenho pode ser pensado 
como uma função da arquitetura entre outras coisas, como os parâmetros, os dados e a duração do treinamento.

![](imagens/redes.jpg)

A base de dados são imagens de rolamentos variados existentes. As imagens não estão em boas condições de fundo, luz,
foco e centralização, um dos primeiros desafios será trabalhar com tratamentos destas imagens através de códigos no Python.
As imagens são de 8 rolamentos distintos, sendo de 03 fotos de cada (frente, verso e lado), 313 fotos totais, 
Conjunto de dados de treino: 270 imagens de rolamentos estão em perfeitas condições Conjunto de dados de validação: 270 imagens
Conjunto de dados de teste: 43 em condições prejudicadas (quebradas, rachadas, com furos, qualidade de imagem ruim), 


![](imagens/black.jpg)

Especificações:

![](imagens/todas.jpg)
![](imagens/4.jpg)

Alguns modelos de rolamentos danificados

![](imagens/df.jpg)


`Tratamento das Imagens`

Conforme especificado acima as imagens encontram-se em estados como problemas de fundo, falta de centralização, 
textura, e imagem colorida, que prejudicam a análise, sendo necessário, como em qualquer base, tratá-las:

`Suavização de imagens`

A suavização da imagem (do inglês Smoothing), também chamada de ‘blur’ ou ‘blurring’ que podemos traduzir para 
“borrão”, é um efeito que podemos notar nas fotografias fora de foco ou desfocadas onde tudo fica embasado. No código 
abaixo percebemos que o método utilizado para a suavização pela média é o método ‘blur’ da OpenCV.

{% highlight python %}
import numpy as np
import cv2
img = cv2.imread('2.jpg') 
img = img[::2,::2] # Diminui a imagem
suave = np.vstack([ 
    np.hstack([img, cv2.blur(img, ( 3, 3))]), 
    np.hstack([cv2.blur(img, (5,5)), cv2.blur(img, ( 7, 7))]), 
    ])
cv2.imshow("Imagens suavisadas (Blur)", suave) 
cv2.imwrite("imagens//blur.jpg", suave)
cv2.waitKey(0)
{% endhighlight %}

![](imagens/blur.jpg)

{% highlight python %}
import cv2
img = cv2.imread('2.jpg')
img_noise = cv2.fastNlMeansDenoisingColored(img, None, 20, 10, 7, 21)
cv2.imshow("img original", img)
cv2.imshow("maqueada", img_noise)
cv2.imwrite("imagens//maqueada1.jpg", img_noise)
cv2.waitKey(0)
{% endhighlight %}

`Escala cinza`
O input da CNN é uma imagem, representada como uma matriz. Cada elemento da matriz contém o valor de seu respectivo pixel, que pode variar de 0 a 255. Para imagens coloridas em RGB, temos uma matriz “em três dimensões”, onde cada dimensão é uma das camadas de cor (red, green e blue). Assim, uma imagem colorida de 255px por 255px é representada por três matrizes de 255 por 255 (255x255x3)
É importante a conversão em escala de cinza, a função imread terá os canais armazenados na ordem BGR (azul, verde e vermelho) por padrão.
Para converter em escala de cinza, necessário chamar a  função cvtColor , que permite converter a imagem de um espaço de cores para outro.
Como primeira entrada, esta função recebe a imagem original. Como segunda entrada, ele recebe o código de conversão do espaço de cores. Como o objetivo é converter a imagem original do espaço de cores BGR para cinza, usamos o código  COLOR_BGR2GRAY .
Para exibir as imagens, precisa chamar o imshow função do cv2 módulo. Esta função recebe como primeira entrada uma string com o nome a ser atribuído à janela e, como segundo argumento, a imagem a ser exibida.
No código pede para exibir as duas imagens para que possa comparar a imagem convertida com a original.
Por fim, a função waitKey, que aguardará um evento do teclado. Esta função recebe como entrada um atraso, especificado em milissegundos. No entanto, se passarmos o valor 0, ele aguardará indefinidamente até que um evento-chave ocorra.

{% highlight python %}
import cv2
img = cv2.imread('2.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("Imagem Original", img)
newimg = cv2.resize(img, (1024,720))
cv2.imshow("Image em escala", gray_img)
cv2.imwrite("imagens//cinza.jpg", gray_img)
cv2.waitKey(0)
{% endhighlight %}

![](imagens/cinza.jpg)

Contudo, existem outros espaços de cores como o próprio “Preto e Branco” ou
 “tons de cinza”, além de outros coloridos como o HSV. Abaixo temos um 
 exemplo de como ficaria nossa imagem da ponte nos outros espaços de cores.

{% highlight python %}
import cv2
img = cv2.imread('2.jpg')
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.imshow("original", img)
cv2.imshow("HSV", img_hsv)
cv2.imwrite("imagens/hsv.jpg", img_hsv)
cv2.waitKey(0)
{% endhighlight %}

![](imagens/hsv.jpg)

Após teste em imagens tratadas de diversas formas o algoritmo conseguiu detectar e reconhecer 
apenas as imagens sem fundo, e com maior performance as que foram aplicadas o efeito Blur.


{% highlight python %}
import cv2
img = cv2.imread('2.jpg')
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.imshow("original", img)
#cv2.imshow("HSV", img_hsv)

lowrgb = (36,0,0)
upperrgb = (255,45,155)
mask1 = cv2.inRange(img_hsv, lowrgb, upperrgb)
mask2= cv2.inRange(img_hsv, (0,0,0), (255,45,155))
end_mask = cv2.bitwise_or(mask1, mask2)
end_img = cv2.bitwise_and(img, img, mask=end_mask)
cv2.imshow("MASCARA", end_img)

cv2.imwrite("imagens\semfundo.jpg", end_img)
cv2.waitKey(0)
{% endhighlight %}

![](imagens/semfundo.jpg)

`Funções organizadas para o tratamento`

{% highlight python %}
# Bibliotecas
import numpy as np
import cv2

# REDIMENSIONAR
def TamanhoImagem(caminhoImagem):
    # Carrega a imagem
    imagem = cv2.imread(caminhoImagem)
    # Redimensiona a imagem para a proporcao passada (0.5 -> Metade do tamando)
    imagem = cv2.resize(imagem, None, fx=0.5, fy=0.5)
    # Sava a nova imagem no caminho escolhido
    cv2.imwrite("novaimagem1.jpg", imagem)

# TONS DE CINZA
def TonsDeCinza(caminhoImagem):
    # Carrega a imagem
    imagem = cv2.imread(caminhoImagem)
    # Converte a imagem para tons de cinza
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    # Sava a nova imagem no caminho escolhido
    cv2.imwrite("novaimagem1.jpg", imagem)

# REMOVER FUNDO
def RemoverFundo(caminhoImagem):
    # Carrega a imagem
    imagem = cv2.imread(caminhoImagem)
    # Remove o fundo da imagem para os parametros passados
    lowrgb = (36,0,0)
    upperrgb = (70,255,255)
    mask1 = cv2.inRange(imagem,lowrgb,upperrgb)
    mask2 = cv2.inRange(imagem,(15,0,0),(36,255,255))
    end_mask = cv2.bitwise_or(mask1,mask2)
    end_img = cv2.bitwise_and(mask1, mask2, mask=end_mask)
    cv2.imshow("Mascara", end_img)
    cv2.waitKey(0)
    # Sava a nova imagem no caminho escolhido
    cv2.imwrite("novaimagem1.jpg", end_img)

# REMOVER RUIDO
def RemoverRuido(caminhoImagem):
    # Carrega a imagem
    imagem = cv2.imread(caminhoImagem)
    # Remove o ruido da imagem
    imagem = cv2.fastNlMeansDenoisingColored(imagem, None,20,10,7,21)
    # Sava a nova imagem no caminho escolhido
    cv2.imwrite("novaimagem1.jpg", imagem)

# CENTRALIZAR IMAGEM
def Centralizar(caminhoImagem):
    # Carrega a imagem
    imagem = cv2.imread(caminhoImagem)
    # Centralizar a imagem
    imagem = cv2.circle(imagem, (1920, 1080), 15, (205, 114, 101), 1)
    # Sava a nova imagem no caminho escolhido
    cv2.imwrite("novaimagem1.jpg", imagem)

# FUNÇÃO PARA AUTOMATIZAR O PROCESSO DE TRATAMENTO
def Tratamento(caminhoImagem):
    TamanhoImagem(caminhoImagem)
    TonsDeCinza(caminhoImagem)
    RemoverFundo(caminhoImagem)
    RemoverRuido(caminhoImagem)
    Centralizar(caminhoImagem)
{% endhighlight %}


`Detecção e reconhecimento de imagens`

{% highlight python %}
import numpy as np 
import cv2 
import mahotas
#Função para facilitar a escrita nas imagem
def escreve(img, texto, cor=(255,0,0)): 
    fonte = cv2.FONT_HERSHEY_SIMPLEX 
    cv2.putText(img, texto, (10,20), fonte, 0.5, cor, 0,
        cv2.LINE_AA)
imgColorida = cv2.imread('imagens//semfundo2.jpg') #Carregamento da imagem
#Se necessário o redimensioamento da imagem pode vir aqui.
#Passo 1: Conversão para tons de cinza
img = cv2.cvtColor(imgColorida, cv2.COLOR_BGR2GRAY)
#Passo 2: Blur/Suavização da imagem 
suave = cv2.blur(img, (7, 7))
#Passo 3: Binarização resultando em pixels brancos e pretos 
T = mahotas.thresholding.otsu(suave) 
bin = suave.copy() 
bin[bin > T] = 255 
bin[bin < 255] = 0 
bin = cv2.bitwise_not(bin)
#Passo 4: Detecção de bordas com Canny 
bordas = cv2.Canny(bin, 70, 150)
#Passo 5: Identificação e contagem dos contornos da imagem 
#cv2.RETR_EXTERNAL = conta apenas os contornos externos 
(lx, objetos, lx) = cv2.findContours(bordas.copy(),
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
#A variável lx (lixo) recebe dados que não são utilizados
escreve(img, "Imagem em tons de cinza", 0) 
escreve(suave, "Suavizacao com Blur", 0) 
escreve(bin, "Binarizacao com Metodo Otsu", 255) 
escreve(bordas, "Detector de bordas Canny", 255) 
temp = np.vstack([ 
    np.hstack([img, suave]), 
    np.hstack([bin, bordas]) 
    ])
cv2.imshow("Quantidade de objetos: "+str(len(objetos)), temp)
cv2.imwrite("imagens//tempBlur2.jpg", temp)
cv2.waitKey(0) 
imgC2 = imgColorida.copy() 
cv2.imshow("Imagem Original", imgColorida)
cv2.drawContours(imgC2, objetos, -1, (255, 0, 0), 2) 
escreve(imgC2, str(len(objetos))+" objetos encontrados!") 
cv2.imshow("Resultado", imgC2) 
cv2.imwrite("imagens//resultadoBlur2.jpg", imgC2)
cv2.waitKey(0)
{% endhighlight %}

Resultados usando blur

![](imagens/blurresult.jpg)

Resultados retirando o fundo

![](imagens/resultadoBlur1.jpg)

Resultados retirando o fundo e ajustando parâmetros

![](imagens/resultadoBlur2.jpg)


`Criação do arquivo XML `

A OpenCV também possui ferramentas para o treinamento do classificador. Primeiro
é necessário criar um arquivo de vetor com o comando opencv_createsamples,
passando a quantidade de exemplos, a largura e a altura como parâmetros. Exemplo:
opencv_createsamples -info anotacoes.txt -num 2400 -w 24 -h 24 -vec positivas_24x24.vec.

Finalmente, o classificador em cascata é treinado com o comando opencv_traincascade,
que recebe como parâmetros um diretório onde o classificador será salvo, o arquivo de vetor
gerado anteriormente, a lista de imagens negativas, a quantidade de imagens positivas e
negativas, o número e estágios da cascata e as dimensões das imagens no vetor. Exemplo:
opencv_traincascade -data classificador/ -vec positivas_24x24.vec -bg negativas.txt -numPos
2000 -numNeg 1000 -numStages 10 -w 24 -h 24.

![](imagens/relogios.jpg)

`Construindo e Treinando o Modelo`

{% highlight python %}
# Imports
import os
import sys
import inspect
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from modulos import utils
from datetime import datetime
from tensorflow.python.framework import ops
from sklearn.metrics.classification import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import warnings
{% endhighlight %}


{% highlight python %}
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("data_dir", "dataset/", "Caminho para o diretório com dados de treino e de teste")
tf.flags.DEFINE_string("logs_dir", "modelo/", "Caminho para o diretório onde o modelo será gravado")
tf.flags.DEFINE_string("mode", "train", "mode: train (Default)/ test")
{% endhighlight %}


{% highlight python %}
def main(argv=None):
    
    # Carrega os dados
    train_images, train_labels, valid_images, valid_labels, test_images = utils.read_data(FLAGS.data_dir)
    
    print("\nTamanho do Dataset de Treino: %s" % train_images.shape[0])
    print('Tamanho do Dataset de Validação: %s' % valid_images.shape[0])
    print("Tamanho do Dataset de Teste: %s" % test_images.shape[0])

    global_step = tf.Variable(0, trainable=False)
    dropout_prob = tf.placeholder(tf.float32)
    input_dataset = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 1], name="input")
    input_labels = tf.placeholder(tf.float32, [None, NUM_LABELS])

    pred = emotionCNN(input_dataset)
    output_pred = tf.nn.softmax(pred, name="output")
    loss_val = loss(pred, input_labels)
    train_op = train(loss_val, global_step)

    summary_op = tf.summary.merge_all()
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Modelo Restaurado!")

        for step in range(MAX_ITERATIONS):
            batch_image, batch_label = get_next_batch(train_images, train_labels, step)
            feed_dict = {input_dataset: batch_image, input_labels: batch_label}

            sess.run(train_op, feed_dict=feed_dict)
            if step % 10 == 0:
                train_loss, summary_str = sess.run([loss_val, summary_op], feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, global_step=step)
                train_error_list.append(train_loss)
                train_step_list.append(step)
                print("Taxa de Erro no Treinamento: %f" % train_loss)

            if step % 100 == 0:
                valid_loss = sess.run(loss_val, feed_dict={input_dataset: valid_images, input_labels: valid_labels})
                valid_error_list.append(valid_loss)
                valid_step_list.append(step)
                print("%s Taxa de Erro na Validação: %f" % (datetime.now(), valid_loss))
                saver.save(sess, FLAGS.logs_dir + 'model.ckpt', global_step=step)
        
        # Plot do erro durante o treinamento
        plt.plot(train_step_list, train_error_list, 'r--', label='Erro no Treinamento Por Iteração', linewidth=4)
        plt.title('Erro no Treinamento Por Iteração')
        plt.xlabel('Iteração')
        plt.ylabel('Erro no Treinamento')
        plt.legend(loc='upper right')
        plt.show()

        # Plot do erro durante a validação
        plt.plot(valid_step_list, valid_error_list, 'r--', label='Erro na Validação Por Iteração', linewidth=4)
        plt.title('Erro na Validação Por Iteração')
        plt.xlabel('Iteração')
        plt.ylabel('Erro na Validação')
        plt.legend(loc='upper right')
        plt.show()  

print(train_error_list) 
print(valid_error_list)
{% endhighlight %}

![](imagens/max.png)

![](imagens/scan.gif)

![](imagens/treino.jpg)

`Teste`

{% highlight python %}

{% endhighlight %}

{% highlight python %}
#Definimos a variável resposta
resultado = {0:'reprovada', 
           1:'aprovada'}
{% endhighlight %}


{% highlight python %}
#colorido para cinza
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
{% endhighlight %}



{% highlight python %}
img = mpimg.imread('images/2.jpg')     
gray = rgb2gray(img)
plt.imshow(gray, cmap = plt.get_cmap('gray'))
plt.show()
{% endhighlight %}

{% highlight python %}
for i in range(0, num_evaluations):
    result = sess.run(y_conv, feed_dict={x:image_0})
    label = sess.run(tf.argmax(result, 1))
    label = label[0]
    label = int(label)
    tResult.evaluate(label)
tResult.display_result(num_evaluations)
{% endhighlight %}

![](imagens/cinza.jpg)


aprovada 98.0% 

{% highlight python %}
img = mpimg.imread('images/rep.jpg')     
gray = rgb2gray(img)
plt.imshow(gray, cmap = plt.get_cmap('gray'))
plt.show()
{% endhighlight %}

{% highlight python %}
for i in range(0, num_evaluations):
    result = sess.run(y_conv, feed_dict={x:image_0})
    label = sess.run(tf.argmax(result, 1))
    label = label[0]
    label = int(label)
    tResult.evaluate(label)
tResult.display_result(num_evaluations)
{% endhighlight %}

![](imagens/rep.jpg)

reprovada 96.0%

{% highlight python %}

{% endhighlight %}


{% highlight python %}

{% endhighlight %}


{% highlight python %}

{% endhighlight %}


{% highlight python %}

{% endhighlight %}


{% highlight python %}

{% endhighlight %}






[jekyll-docs]: https://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/
