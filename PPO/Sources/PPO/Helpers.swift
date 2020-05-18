import TensorFlow

struct PPOHelpers {
    @differentiable
    static func logLikelihood(x: Tensor<Float>, mu: Tensor<Float>, logStd: Tensor<Float>) -> Tensor<Float> {
        let std = exp(logStd) + 1e-8
        let logPi = Tensor<Float>(log(2.0*Float.pi))
        let n = (x - mu) / std
        return -0.5 * (pow(n, Tensor<Float>(2.0)) + 2.0 * logStd + logPi)
    }
    
    @differentiable(wrt: logP)
    static func policyLoss(oldLogP: Tensor<Float>, logP: Tensor<Float>, adv: Tensor<Float>, clipVal: Float = 0.2) -> Tensor<Float> {
        let ratio = exp(logP - oldLogP)
        let minAdv = ratio.clipped(min: 1.0-clipVal, max: 1.0+clipVal) * adv
        return -(min(ratio * adv, minAdv)).mean()
    }
    
    static func approxKL(oldLogP: Tensor<Float>, logP: Tensor<Float>) -> Float {
        return (oldLogP - logP).mean().scalarized()
    }
}

struct PPOReplayMemory {
    
    private static func gaeLambda(v: Tensor<Float>, r: Tensor<Float>, gamma: Float = 0.99, lambda: Float = 0.97) -> Tensor<Float> {
        let length = r.shape[0] - 1
        let adv = r[0..<length] + gamma * v[1..<length+1] - v[0..<length]
        return discount(adv, gamma: gamma * lambda)
    }
    
    private static func discount(_ r: Tensor<Float>, gamma: Float = 0.99) -> Tensor<Float> {
        var dr = r[r.shape[0]-1].rankLifted()
        for i in (0..<r.shape[0]-1).reversed() {
            dr = (r[i] + gamma * dr[0]).rankLifted().concatenated(with: dr)
        }
        return dr
    }
    
    struct Step {
        let o: Tensor<Float>
        let a: Tensor<Float>
        let oldLogP: Tensor<Float>
        let r: Tensor<Float>
        let v: Tensor<Float>
    }
    
    private let batchShape: TensorShape
    private let obsBatchShape: TensorShape
    private var trajs = [PPOReplayMemory.Step]()
    
    init(steps: Int, n: Int, obsShape: TensorShape) {
        self.batchShape = TensorShape([steps * n, -1])
        self.obsBatchShape = TensorShape([-1] + obsShape.dimensions)
    }
    
    mutating func append(steps: [Step], lastVal: Tensor<Float>) {
        let o = Tensor<Float>(steps.map { $0.o })
        let a = Tensor<Float>(steps.map { $0.a })
        let oldLogP = Tensor<Float>(steps.map { $0.oldLogP })
        var r = Tensor<Float>(steps.map { $0.r } + [lastVal])
        let v = Tensor<Float>(steps.map { $0.v } + [lastVal])
        let adv = PPOReplayMemory.gaeLambda(v: v, r: r)
        r = PPOReplayMemory.discount(r)[0..<r.shape[0]-1]
        trajs.append(PPOReplayMemory.Step(o: o, a: a, oldLogP: oldLogP, r: r, v: adv))
    }
    
    func getBatch() -> (Tensor<Float>, Tensor<Float>, Tensor<Float>, Tensor<Float>, Tensor<Float>) {
        // <- concatenating initializer N == 1 crash ->
        if trajs.count == 1 {
            let adv = trajs[0].v.reshaped(to: batchShape)
            let normAdv = (adv - adv.mean(alongAxes: 0)) / adv.variance(alongAxes: 0)
            return (trajs[0].o.reshaped(to: obsBatchShape),
                    trajs[0].a.reshaped(to: batchShape),
                    trajs[0].oldLogP.reshaped(to: batchShape),
                    trajs[0].r.reshaped(to: batchShape),
                    normAdv)
        }
        
        let o = Tensor<Float>(concatenating: trajs.map { $0.o }, alongAxis: 0).reshaped(to: obsBatchShape)
	    let a = Tensor<Float>(concatenating: trajs.map { $0.a }, alongAxis: 0).reshaped(to: batchShape)
	    let oldLogP = Tensor<Float>(concatenating: trajs.map { $0.oldLogP }, alongAxis: 0).reshaped(to: batchShape)
	    let r = Tensor<Float>(concatenating: trajs.map { $0.r }, alongAxis: 0).reshaped(to: batchShape)
	    let adv = Tensor<Float>(concatenating: trajs.map { $0.v }, alongAxis: 0).reshaped(to: batchShape)
        let normAdv = (adv - adv.mean(alongAxes: 0)) / adv.variance(alongAxes: 0)
        return (o, a, oldLogP, r, normAdv)
    }
}
