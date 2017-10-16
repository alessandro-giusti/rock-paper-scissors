# Dialogue En Web

Dialogue En Web è un servizio web commissionato da Iras Cotis per il progetto Dialogue en Route: esso permette di cercare le "stazioni" ed i "percorsi", luoghi geografici facenti parte del progetto
Le stazioni sono singoli luoghi di interesse, mentre i percorsi si compongono di più stazioni.
L'utente ha la possibilità di vedere entrambi su una mappa, e di ottenere informazioni aggiuntive su di essi.
Un ipotetico admin ha la possibilità di modificare il contenuto dei database.

<br>
Il servizio si compone delle seguenti chiamate:
<ul>
	<li>http://isin03.dti.supsi.ch:9090/DIGNAZIO_FEDERICO/stazione?nomeStazione=Krishna-Gemeinschaft</li> GET
	<li>http://isin03.dti.supsi.ch:9090/DIGNAZIO_FEDERICO/stazione/5</li> GET
	<li>http://isin03.dti.supsi.ch:9090/DIGNAZIO_FEDERICO/stazione/5</li> DELETE
	<li>http://isin03.dti.supsi.ch:9090/DIGNAZIO_FEDERICO/stazione</li> POST
	<li>http://isin03.dti.supsi.ch:9090/DIGNAZIO_FEDERICO/stazione/all</li> GET
	<li>http://isin03.dti.supsi.ch:9090/DIGNAZIO_FEDERICO/stazione/info?nomeStazione=Krishna-Gemeinschaft</li> GET


    <li>http://isin03.dti.supsi.ch:9090/DIGNAZIO_FEDERICO/percorso?name=Moschee</li> GET
	<li>http://isin03.dti.supsi.ch:9090/DIGNAZIO_FEDERICO/percorso/5</li> DELETE
	<li>http://isin03.dti.supsi.ch:9090/DIGNAZIO_FEDERICO/percorso</li> POST
	<li>http://isin03.dti.supsi.ch:9090/DIGNAZIO_FEDERICO/percorso/all</li> GET

</ul>

<br>
Il client è raggiungibile al seguente indirizzo:
<ul>
	<li>http://isin03.dti.supsi.ch:9090/DIGNAZIO_FEDERICO</li>
</ul>

<br>
Per informazioni aggiuntive al seguente link è presente la Doc in formato PDF:
<ul>
	<li>http://isin03.dti.supsi.ch:9090/DIGNAZIO_FEDERICO/docs/doc.pdf</li>
</ul>

