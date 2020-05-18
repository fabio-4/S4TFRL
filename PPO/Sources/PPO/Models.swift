import TensorFlow

struct PPOPolicy: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>
    
    var layer1: Dense<Float>
    var layer2: Dense<Float>
    var layer3: Dense<Float>
    var logStd: Tensor<Float>
    
    init(inputSize: Int, outputSize: Int) {
        layer1 = Dense<Float>(inputSize: inputSize, outputSize: 128, activation: tanh)
        layer2 = Dense<Float>(inputSize: 128, outputSize: 128, activation: tanh)
        layer3 = Dense<Float>(inputSize: 128, outputSize: outputSize)
        logStd = Tensor<Float>(repeating: -0.5, shape: TensorShape(outputSize))
    }
    
    @differentiable
    func callAsFunction(_ input: Input) -> Output {
        return input.sequenced(through: layer1, layer2, layer3)
    }

    func callAsFunction(_ input: Input) -> (Output, Output) {
        let mu: Output = self(input)
        let a = mu + exp(logStd) * _Raw.randomStandardNormal(shape: mu.shapeTensor)
        let logA = PPOHelpers.logLikelihood(x: a, mu: mu, logStd: logStd).sum(alongAxes: -1)
        return (a, logA)
    }
}

struct PPOValue: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>
    
    var layer1: Dense<Float>
    var layer2: Dense<Float>
    var layer3: Dense<Float>
    
    init(inputSize: Int) {
        layer1 = Dense<Float>(inputSize: inputSize, outputSize: 128, activation: tanh)
        layer2 = Dense<Float>(inputSize: 128, outputSize: 128, activation: tanh)
        layer3 = Dense<Float>(inputSize: 128, outputSize: 1)
    }
    
    @differentiable
    func callAsFunction(_ input: Input) -> Output {
        return input.sequenced(through: layer1, layer2, layer3)
    }
}
