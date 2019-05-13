from keras.applications import InceptionResNetV2


def main():
  inception_model = InceptionResNetV2()

  for layer in inception_model.layers:
    print(layer.name)
    layer.trainable = False

if __name__ == '__main__':
  main()