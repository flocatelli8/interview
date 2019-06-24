#' ---
#' title: "Interview"
#' author: "Francesco Locatelli"
#' output: 
#'   html_document: 
#'     fig_width: 9
#'     df_print: kable
#'     toc: yes
#'     number_sections: true
#' ---
#' 
## ----setup, include=FALSE------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)

#' 
#' # Libraries
#' 
#' Carico le librerie
## ---- message=FALSE------------------------------------------------------
# Import libraries
pkg <- installed.packages()

if (!"DBI" %in% pkg[,1]) install.packages("DBI")
if (!"RPostgres" %in% pkg[,1]) install.packages("RPostgres")
if (!"dplyr" %in% pkg[,1]) install.packages("dplyr")
if (!"plotly" %in% pkg[,1]) install.packages("plotly")
if (!"caret" %in% pkg[,1]) install.packages("caret")
if (!"e1071" %in% pkg[,1]) install.packages("e1071")
if (!"glmnet" %in% pkg[,1]) install.packages("glmnet")
if (!"randomForest" %in% pkg[,1]) install.packages("randomForest")

require(DBI)
require(RPostgres)
require(dplyr)
require(plotly)
require(caret)

#' 
#' # Import data
#' Connessione al db
## ---- message=FALSE------------------------------------------------------
# Connecting to db
con = dbConnect(
  Postgres(), 
  user = 'my_user',
  password = 'my_pssw',
  dbname = 'my_db',
  host = 'my_host',
  port = 25060,
  sslmode = 'require'
)

#' 
#' Importo i dati in R
## ---- message=FALSE------------------------------------------------------
regressors_query <- dbSendQuery(con, "SELECT * FROM ccpp.regressors")
regressors <- dbFetch(regressors_query)
dbClearResult(regressors_query)

target_query <- dbSendQuery(con, "SELECT * FROM ccpp.target")
target <- dbFetch(target_query)
dbClearResult(target_query)

# Closing connection
dbDisconnect(con)

#' 
#' 
#' # Esplorazione dati
#' 
#' ## Dimensioni e intestazione:
## ---- message=FALSE------------------------------------------------------
head(regressors)
head(target)
dim(target)
dim(regressors)

#' 
#' ## Join
#' Merge/join delle tabelle regressors e target attraverso la chiave univoca ID:
## ----message=FALSE-------------------------------------------------------
data <- dplyr::inner_join(regressors, target, by = 'id')
head(data)

#' 
#' 
#' ## Analisi
#' Analisi delle variabili covariate e target, distribuzione e valori mancanti (missing value):
## ---- message=FALSE------------------------------------------------------
summary(data[,-1])

#' Le variabili _at_ e _pe_ contengono 1 valore mancante ciascuna. 
#' Le variabili _ap_ e _pe_ hanno un valore massimo decisamente superiore alma media e 3° quartile: analizziamone la distribuzione per determinare la possibile presenza di outlier.
#' 
## ---- message=FALSE, fig.height=4----------------------------------------
par(mfrow=c(1,2))
p1<-plot(data$ap, main = 'Plot ap')
p2<-hist(data$ap, main = 'Histogram ap')
p1<-plot(data$pe, main = 'Plot pe')
p2<-hist(data$pe, main = 'Histogram pe')
par(mfrow=c(1,1))

#' Dai grafici deduco che il regressore _ap_ e la variabile dipendente _pe_ hanno rispettivamente un valore anomalo _outlier_: decido di eslcudere questi valori dal dataset che utilizzerò per il modello.
#' 
#' Rimozione valori anomali e missing:
## ---- message=FALSE------------------------------------------------------
pos_NA <- c(which(is.na(data$at)),which(is.na(data$pe)))
pos_outlier <- c(which.max(data$ap),which.max(data$pe))

data_clean <- data[-c(pos_NA, pos_outlier),-1]
summary(data_clean)

#' 
#' Matrice di correlazione di Pearson
## ------------------------------------------------------------------------
cor(data_clean)

#' 
#' L'indice di correlazione suggerisce che sussiste un grado di dipendeza, talvolta forte, dei regressori con la variabile dipendente _pe_. La variabile più correlata con il target è il regressore _at_, tramite una forte dipendenza negativa.
#' 
#' # Plot
#' 
#' Scatter plot dei regressori vs variabile dipendente:
## ---- message=FALSE, fig.align='center'----------------------------------
plot_ly(data_clean, x=~at, y=~pe, mode = 'markers', type = 'scatter')
plot_ly(data_clean, x=~v, y=~pe, mode = 'markers', type = 'scatter')
plot_ly(data_clean, x=~ap, y=~pe, mode = 'markers', type = 'scatter')
plot_ly(data_clean, x=~rh, y=~pe, mode = 'markers', type = 'scatter')

#' 
#' # Models
#' 
#' ## Variable selection
#' 
#' Dato che le covariate sono `r ncol(data_clean)-1` e le osservazioni `r nrow(data_clean)`, i gradi di libertà dei modelli saranno elevati. Di conseguenza non risulterà necessario effettuare un processo di variable selection.
#' 
#' ## Model selection
#' 
#' ### Intro
#' 
#' Dal momento che la variabile dipendente _pe_ è quantitativa continua, il problema di fitting è riconducile alla classe dei modelli supervisionati applicabili a contesti di regressione.
#' 
#' ### Cross-Validation
#' 
#' Pirma di testare i modelli è necessario:
#' * definire la __metrica da minimizzare__ e utilizzare per confrontare il fit del modelli: __RMSE__
#' * dotarsi di un metodo di _train/test_ che consenta di evitare il problema dell'overfitting e verificare che il modello si comporti al meglio in contesti predittivi oltre che esplicativi.
#' 
#' Scelgo di applicare una _k-cross-validation_ (non ripetuta) con 5 _folds_: il dataset sarà diviso in 5 gruppi ed il modello sarà stimato (train) su 4 gruppi (80%) e poi applicato (test) su di 1 gruppo (20%). COn questo metodo tutte le osservazioni sono utilizzate sia per stimare il modello che per controllare il fit.
## ------------------------------------------------------------------------
fitControl <- trainControl(method = "cv", number = 5)

#' 
#' 
#' ### Basic linear model
#' 
#' Come primo step vediamo come si comporta la regressione lineare multipla: questo modello sarà usato come benchmark per valutare le performance di modelli di ML più sofisticati:
#' 
## ---- message=FALSE------------------------------------------------------
set.seed(8848)
lm1 <- train(pe~., data = data_clean, trControl=fitControl, method = 'lm')
lm1

#' 
#' ### ML Methods
#' 
#' __Generalized Additive Model__
## ---- message=FALSE------------------------------------------------------
# Could be slow
set.seed(8848)
gam1 <- train(pe~., data = data_clean, trControl=fitControl, method = 'gam')
gam1

#' 
#' __Boosted Generalized Additive Model__
## ---- message=FALSE, warning=FALSE---------------------------------------
set.seed(8848)
# Could be slow
gamboost1 <- train(pe~., data = data_clean, trControl=fitControl, method = 'gamboost')
gamboost1

#' 
#' __Lasso regression__
## ---- message=FALSE------------------------------------------------------
set.seed(8848)
lasso1 <- train(pe~., data = data_clean, trControl=fitControl, method = 'lasso')
lasso1

#' 
#' __Ridge regression__
## ---- message=FALSE------------------------------------------------------
set.seed(8848)
ridge1 <- train(pe~., data = data_clean, trControl=fitControl, method = 'ridge')
ridge1

#' 
#' 
#' __Polynomial Regression__
#' Faccio variare i gradi del polinomio
## ---- message=FALSE------------------------------------------------------
# Could be slow
model_poly <- as.data.frame(matrix(0, ncol = 5, nrow = 0))
names(model_poly) <- c("i","j","k","h","RMSE")
r<-1

for(i in 1:3) {
  for(j in 1:3){
    for(k in 1:3){
      for(h in 1:3){
  f_name <-paste0("pe~poly(at,",i,")+poly(v,",j,")+poly(ap,",k,")+poly(rh,",h,")")
  form <- formula(f_name)
  set.seed(8848)
  model <- train(form, data = data_clean, trControl=fitControl, method = 'lm')
  model_poly[r,"i"] <- i
  model_poly[r,"j"] <- j
  model_poly[r,"h"] <- h
  model_poly[r,"k"] <- k
  model_poly[r,"RMSE"] <- model$results$RMSE
  r <- r+1
      }
    }
  }
}

model_poly[which.min(model_poly$RMSE),]

#' 
#' __Linear Model with Interactions__
## ---- message=FALSE------------------------------------------------------
set.seed(8848)
int1 <- train(pe~at*v*ap*rh, data = data_clean, trControl=fitControl, method = 'lm')
int1

#' 
#' __Random Forest__
#' Con CV il modello non è ottimizzato: utilizzo un approccio train-test 80-20 classico.
## ---- message=FALSE------------------------------------------------------
# Could be slow
seq <- seq(1,nrow(data_clean),1)
s <- sample(seq,as.integer(nrow(data_clean)*0.8))
data_train_rf <- data_clean[s,]
data_test_rf <- data_clean[-s,]

set.seed(8848)
rf1 <- randomForest::randomForest(pe~., data = data_train_rf, ntree = 200, mtry = 3, importance = TRUE) 
data_test_rf$predict <-predict(rf1, data_test_rf)
data_test_rf$residuals <- data_test_rf$predict - data_test_rf$pe
rf_RMSE <- sqrt(mean((data_test_rf$residuals)^2))

#' 
#' ### Final Model
#' 
#' Scelgo il modello che minimizza RMSE:
## ------------------------------------------------------------------------
model_results <- as.data.frame(rbind(
  cbind(lm1$results$RMSE, "lm1","linear model"),
  cbind(gamboost1$results$RMSE, "gamboost1", "gamboost"),
  cbind(gam1$results$RMSE, "gam1", "gam"),
  cbind(lasso1$results$RMSE, "lasso1","lasso"),
  cbind(ridge1$results$RMSE, "ridge1","ridge"),
  cbind(model_poly[which.min(model_poly$RMSE),"RMSE"], "model_poly", "Polynomial lm"),
  cbind(int1$results$RMSE, "int1", "Lm with interactions"),
  cbind(rf_RMSE, "rf1", "random forest")
))
names(model_results) <- c("RMSE", "model", "type")
model_results$RMSE <- gsub(" ", "",model_results$RMSE)
pos_best <- which.min(as.numeric(model_results$RMSE))
model_results[pos_best,]

#' 
#' Summary del modello finale:
## ------------------------------------------------------------------------
get(as.character(model_results[pos_best,"model"]))

#' 
#' Importanza delle variabili:
## ------------------------------------------------------------------------
get(as.character(model_results[pos_best,"model"]))$importance

#' 
#' ## Conclusioni
#' Ho scelto il modello che minimizza RMSE. Avendo utilizzato un approccio di model selection robusto, non incorro in problemi di _overfitting_. L'tilizzo di un modello di ML migliora le performance predittive del `r round((lm1$results$RMSE-rf_RMSE)/lm1$results$RMSE,2)`% rispetto al benchmark (regressione lineare multipla), che già godeva di un buon fit ( R-squared = `r lm1$results$Rsquared`)
