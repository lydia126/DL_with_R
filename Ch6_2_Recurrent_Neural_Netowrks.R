# Deep learing with R Chapter 6
# Local : D:\GitHub\DL_with_R
# Remote: https://github.com/lydia126/DL_with_R

# 6.2 Understanding recurrent neural networks
# 6.2.1 A recurrent layer in Keras

# Listing 6.22 Preparing the IMDB data
library(keras)
max_features <- 10000
maxlen <- 500
batch_size <- 32
cat("Loading data...\n")
imdb <- dataset_imdb(num_words = max_features)
c(c(input_train, y_train), c(input_test, y_test)) %<-% imdb
cat(length(input_train), "train sequences\n")
cat(length(input_test), "test sequences")

cat("Pad sequences (samples x time)\n")
input_train <- pad_sequences(input_train, maxlen = maxlen)
input_test <- pad_sequences(input_test, maxlen = maxlen)
cat("input_train shape:", dim(input_train), "\n")
cat("input_test shape:", dim(input_test), "\n")

# Listing 6.23 Training the model with embedding and simple RNN layers
model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_features, output_dim = 32) %>%
  layer_simple_rnn(units = 32) %>%
  layer_dense(units = 1, activation = "sigmoid")

### 에러 발생> reticulate::py_config() 확인 후 numpy version 1.20.2 -> 1.19.5로 
### anaconda shell에서 pip install numpy==1.19.5

model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)

history <- model %>% fit(
  input_train, y_train,
  epochs = 10,
  batch_size = 128,
  validation_split = 0.2
)

plot(history)

#6.2.2 Understanding the LSTM and GRU layers
#6.2.3 A concrete LSTM example in Keras
model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_features, output_dim = 32) %>%
  layer_lstm(units = 32) %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)

history <- model %>% fit(
  input_train, y_train,
  epochs = 10,
  batch_size = 128,
  validation_split = 0.2
)

