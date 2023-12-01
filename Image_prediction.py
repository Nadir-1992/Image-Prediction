from imageai.Classification import ImageClassification

prediction = ImageClassification()

prediction.setModelTypeAsMobileNetV2()
prediction.setModelPath("data.pth")
prediction.loadModel()

predictions, probabilities = prediction.classifyImage("car.jpg", result_count=3)
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction, " : ", eachProbability)
