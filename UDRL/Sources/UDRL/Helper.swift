import TensorFlow

struct BehaviorModel: Layer {
    //use Input = (Tensor<Float>, Tensor<Float>, Tensor<Float>), first function unused
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>
    
    var layer1: Dense<Float>
    var layer2: Dense<Float>
    var layer3: Dense<Float>
    
    init(inputSize: Int, outputSize: Int) {
        layer1 = Dense<Float>(inputSize: inputSize + 2, outputSize: 32, activation: relu)
        layer2 = Dense<Float>(inputSize: 32, outputSize: 64, activation: relu)
        layer3 = Dense<Float>(inputSize: 64, outputSize: outputSize)
    }
    
    @differentiable
    func callAsFunction(_ input: Input) -> Output {
        return input.sequenced(through: layer1, layer2, layer3)
    }
    
    @differentiable
    func callAsFunction(_ input: Input, _ dh: Input, _ dr: Input) -> Output {
        return self(input.concatenated(with: dh * 0.05, alongAxis: -1)
            .concatenated(with: dr * 1.0, alongAxis: -1))
    }
}

struct Episode {
    struct Step {
        let o: Tensor<Float>
        let a: Tensor<Int32>
        let r: Tensor<Float>
    }
    let steps: [Step]
    let r: Float
}

extension Array where Element == Episode {
    mutating func insertSorted(_ e: Episode, maxLen: Int = 50) {
        let index = self.firstIndex(where: { e.r >= $0.r }) ?? self.endIndex
        if index < maxLen {
            self.insert(e, at: index)
            if self.count > maxLen { self = self.dropLast() }
        }
    }
    
    func getCommand(len: Int = 15) -> (Tensor<Float>, Tensor<Float>) {
        let endi = Swift.min(len, self.endIndex)
        if endi < 1 {
            return (
                Tensor<Float>(shape: [1, 1], scalars: [1.0]),
                Tensor<Float>(shape: [1, 1], scalars: [1.0])
            )
        }
        let r = Tensor<Float>(self[0..<endi].map { $0.r })
        let hmean = Tensor<Float>(self[0..<endi].map { Float($0.steps.count) }).mean()
        let rmean = r.mean().scalarized()
        let rstd = r.standardDeviation().scalarized()
        return (
            hmean.reshaped(to: [1, 1]),
            Tensor<Float>(shape: [1, 1], scalars: [Float.random(in: rmean..<rmean+rstd+1e-5)])
        )
    }
    
    func sample(_ n: Int = 32) -> (Tensor<Float>, Tensor<Float>, Tensor<Float>, Tensor<Int32>) {
        var obss = [Tensor<Float>]()
        var dhs = [Tensor<Float>]()
        var drs = [Tensor<Float>]()
        var targets = [Tensor<Int32>]()
        for _ in 0..<n {
            if let e = self.randomElement() {
                let t0 = Int.random(in: 0..<e.steps.count)
                let t1 = e.steps.count
                obss.append(e.steps[t0].o)
                dhs.append(Tensor<Float>([Float(t1-t0)]))
                let r = e.steps[t0..<t1].reduce(Tensor<Float>(0), { $0 + $1.r })
                drs.append(r)
                targets.append(e.steps[t0].a)
            }
        }
        return (Tensor<Float>(obss), Tensor<Float>(dhs), Tensor<Float>(drs), Tensor<Int32>(targets))
    }
}
