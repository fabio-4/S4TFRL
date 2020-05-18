import TensorFlow

extension Layer {
    mutating func softUpdate(from source: Self, tau: Float = 0.01) {
        for kp in self.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
            self[keyPath: kp] = (1.0-tau) * self[keyPath: kp] + tau * source[keyPath: kp]
        }
    }
}

struct DQNEpisode {
    struct Step {
        let o1: Tensor<Float>
        let o2: Tensor<Float>
        let a: Tensor<Int32>
        let r: Tensor<Float>
        let d: Tensor<Bool>
    }
    let steps: [Step]
    
    func getBatch() -> (Tensor<Float>, Tensor<Float>, Tensor<Int32>, Tensor<Float>, Tensor<Float>) {
        let o1 = Tensor<Float>(steps.map { $0.o1 })
        let o2 = Tensor<Float>(steps.map { $0.o2 })
        let a = Tensor<Int32>(steps.map { $0.a })
        let r = Tensor<Float>(steps.map { $0.r })
        let d = Tensor<Float>(steps.map { Tensor<Float>($0.d) })
        return (o1, o2, a, r, d)
    }
}
