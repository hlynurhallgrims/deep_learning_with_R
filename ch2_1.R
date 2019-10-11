# Hlaða -------------------------------------------------------------------
library(keras)

# Lesa --------------------------------------------------------------------
mnist <- dataset_mnist()
þjálfun_myndir    <- mnist$train$x
þjálfun_merkingar <- mnist$train$y
prófun_myndir     <- mnist$test$x
prófun_merkingar  <- mnist$test$y

#Skoða mynd

for (i in 1:15) {

  tala <- þjálfun_myndir[i, , ]
  plot(as.raster(tala, max = 255))

}
# Hanna net ---------------------------------------------------------------
tauganet <- keras_model_sequential() %>% 
  layer_dense(units = 512, activation = "relu", input_shape = c(28 * 28)) %>% 
  layer_dense(units = 10, activation = "softmax")

  # Samsetja net ------------------------------------------------------------
tauganet %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

# Undirbúa ----------------------------------------------------------------
#Gögn
þjálfun_myndir <- array_reshape(þjálfun_myndir, c(60000, 28 * 28))
þjálfun_myndir <- þjálfun_myndir / 255
prófun_myndir <- array_reshape(prófun_myndir, c(10000, 28 * 28))
prófun_myndir <- prófun_myndir / 255
#Merkingar  
þjálfun_merkingar <- to_categorical(þjálfun_merkingar)
prófun_merkingar <- to_categorical(prófun_merkingar)

# Þjálfa ------------------------------------------------------------------
tauganet %>% fit(þjálfun_myndir, þjálfun_merkingar, epochs = 5, batch_size = 128)

# Meta --------------------------------------------------------------------
mæling <- tauganet %>% evaluate(prófun_myndir, prófun_merkingar)
mæling

tala <- þjálfun_myndir[5,,]
þjálfun_myndir
