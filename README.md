# CLASIFICACIÓN DE IMÁGENES

Con este proyecto nos interesa probar las diferentes técnicas de la analítica para poder realizar una clasificación de imágenes de nuestro dataset, donde se encuentran imágenes de dos vegetales: el culantro y el perejil; ya que, incluso hoy en día se hace complicado diferenciar ambos, sobretodo a las personas que no están familiarizadas con la cocina.

CULANTRO

![Culantro](https://github.com/davidcole041095/Analitica/blob/master/culantro.jpg)   



PEREJIL

![Perejil](https://github.com/davidcole041095/Analitica/blob/master/perejil.jpg)

Como se puede apreciar en las dos imágenes anteriores, ambos vegetales, son similares, por lo que causa una confusión al momento de identificarlas.

Además generamos una curva ROC para poder obtener un AUC de nuestro modelo

"Imagen curva ROC y AUC"

* Usamos la validación del 30% de nuestro modelo y un train del 70% para la clasificacón de imágenes de nuestro dataset	
* Usamos las siguientes técnicas: 
	* Regresión logística
	* Análisis discriminante
	* Clasificación de imágenes
	* Árbol de clasificación
	* Naive Bayer
	* SVC Suport vector machine cassifier

El trabajo fue realizado en Google Colab:

https://colab.research.google.com/drive/1y6IhtIqPAYpd5i_sJonM_3fhmRZ8xm1l?authuser=2#scrollTo=5Qlszsl59JEY
