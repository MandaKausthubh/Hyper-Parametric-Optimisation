

class PBTTuner:
    
    def __init__(self, model, NumberOfAgents, CutOffMeasure, NumberOfLearningIterations, InitialHyperParameter, loss='sparse_categorical_crossentropy'):
        self.listOfAgents = [keras.models.clone_model(model) for _ in range(NumberOfAgents)]
        self.HyperParameters = InitialHyperParameter
        self.NumberOfAgents = NumberOfAgents
        self.InitialHyperParameter = InitialHyperParameter
        self.NumberOfLearningIterations = NumberOfLearningIterations
        self.CutOffMeasure = CutOffMeasure
        self.Iterations = NumberOfLearningIterations
        self.loss = loss
        for Agent, rate in zip(self.listOfAgents,self.HyperParameters):
            optimizer = keras.optimizers.Adam(learning_rate=rate)
            Agent.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        
    def PBTOptimizeFit(self, train_data, x_data, y_data, numDataPoints = 1000):
        # 1. Calculate the size of the data slices
        sliceSize = int(len(x_data)/numDataPoints)

        # 2. Create a list of indices
        accuracy_val = []
        loss_val = []
        for iter in range(self.NumberOfLearningIterations):
            index = iter % (len(x_data) // sliceSize)
            x = x_data[index * sliceSize:(index + 1) * sliceSize]
            y = y_data[index * sliceSize:(index + 1) * sliceSize]
            for Agent in self.listOfAgents:
                Agent.fit(x, y, epochs=1)
            losses =np.zeros(self.NumberOfAgents)
            for i, Agent in enumerate(self.listOfAgents):
                loss, accuracy = Agent.evaluate(x, y)
                losses[i] = loss
                accuracy_val.append(accuracy)
                loss_val.append(loss)
            bestAgent = np.argmin(losses)
            worstAgents = np.argsort(losses) [self.CutOffMeasure:]
            for i in worstAgents:
                noisyweights = [
                    np.random.normal(0, 0.01, w.shape) + w
                    for w in self.listOfAgents[bestAgent].get_weights()
                ]
                self.HyperParameters[i] = self.HyperParameters[bestAgent]
                self.listOfAgents[i].set_weights(noisyweights)
                optimizer = keras.optimizers.Adam(learning_rate=self.HyperParameters[bestAgent] + np.random.normal(0, 0.000001))
                self.listOfAgents[i].compile(optimizer=optimizer, loss=self.loss, metrics=['accuracy'])

        plt.plot(accuracy_val)
        plt.plot(loss_val)
        
        losses = np.zeros(self.NumberOfAgents)
        for i, Agent in enumerate(self.listOfAgents):
            losses[i] = Agent.evaluate(x_data, y_data)[0]
        bestAgent = np.argmin(losses)
        return self.listOfAgents[bestAgent]
            
