# Neuronske mreže 2018/2019

U projektu je ispitano kako različite afine transformacije utiču prilikom treniranje i testiranja sijamskih neuronskih mreža.
Za implementaciju je korišten Tensorflow 1 i Keras.

Obrađena su četiri slučaja: treniranje i testiranje bez transformacija, treniranje i testiranje sa transformacijama, treniranje bez transformacija, a testiranje sa transformacijama te konačno, treniranje sa transformacijama i testiranje bez transformacija. 

Da bi se trenirali modeli sa i bez transformacija, pozvati **train_models.py**.

Za određivanje pozitivnih i negativnih parova, pozvati **identifying_images.py**. Potrebno je prvo postaviti promenljivu **model_location** na direktorijum u kojemu se nalaze modeli trenirani preko train_models.py ('./data/...')

Za dobijanje tabelarnog prikaza rezultata, pokrenuti **get_tables.py**.

Pošto je prilikom učitavanja transformacija za cifru 4 uočena najveće odstupanje, njeni podaci su detaljnije obrađeni. Da bi se dobio "embedding" za tu cifru, potrebno je pokrenuti **projecting_4.py**, te nakon toga **get_plots.py** nakon čega će biti sačuvani grafovi sa označenim pozitivnim i negativnim rezultatima.

U **tables.pdf** se nalazi tabelarni prikaz rezultata, sa izračunatim vrednostima za precision, recall i F1 score sa izračunatom standardnom devijacijom.
