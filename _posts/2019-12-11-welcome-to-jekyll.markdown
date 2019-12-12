---
layout: post
title:  "Visão computacional para indústrias"
date:   2019-12-11 15:39:56 -0300
categories: jekyll update
---

Atualmente, as indústrias possuem muitos meios ainda manuais para inspeção e verificação de 
 peças, a fim de garantir que o produto fabricado esteja em condições de funcionalidade e 
 garantia de qualidade. Para isso são necessários funcionários específicos para esta atividade,
 na qual precisam analisar cada peça e retirar peças que não entram dentro da legislação e 
 qualificação. Devido a intensa quantidade de produção, torna-se um trabalho extremamente 
 cansativo e passível a muitas falhas. O objetivo deste artigo é trazer uma solução para 
 este serviço tão manual, que seria substituir as etapas de análise e verificação das 
 inspeções das peças por um modelo de Deep Learning através de uma imagem e o modelo terá 
 que identificar se a imagem possui anormalidades na peça.

![](imagens/black.jpg)

Abaixo a descrição dos rolamentos

![](imagens/tabrol.png)

Deste total 43 rolamentos foram prejudicadas e modificados para que o programa diferencie 
com as demais.

![](imagens/def.jpg)


`Tratamento das Imagens`

1. Suavização de imagens

A suavisação da imagem (do inglês Smoothing), também chamada de ‘blur’ ou ‘blurring’ que 
podemos traduzir para “borrão”, é um efeito que podemos notar nas fotografias fora de foco 
ou desfocadas onde tudo fica embasado. No código abaixo percebemos que o método utilizado 
para a suavização pela média é o método ‘blur’ da OpenCV.

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

Em alguns casos é necessario usar a imagem no tom de cinza

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
 “tons de cinza”, além de outros coloridos como o L*a*b* e o HSV. Abaixo temos um 
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

Funções organizadas para o tratamento

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


[jekyll-docs]: https://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/
