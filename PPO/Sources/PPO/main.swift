import TensorFlow
import S4TFUnityGym

let path = "/...PATHTO/3DBall.app"
guard let env = try? UnityGym(path) else {
    exit(0)
}
defer { env.close() }

let nO = env.numStackedObservations
let inputLen = env.observationSpace.shape.contiguousSize * nO
let actionDim = env.actionSpace.shape[0]
let n = env.nAgents

var policy = PPOPolicy(inputSize: inputLen, outputSize: actionDim)
var value = PPOValue(inputSize: inputLen)
let pOpt = Adam(for: policy, learningRate: 3e-4)
let vOpt = Adam(for: value)
let targetKL: Float = 0.015
var memory: PPOReplayMemory!
Context.local.learningPhase = .training

private func runEpisode(maxSteps: Int) throws -> Int {
    var episodeReward: Float = 0.0
    var steps = [PPOReplayMemory.Step]()
    var o1 = try env.reset()
    var done = Tensor<Bool>(repeating: false, shape: [n])
    var t = 0
    while t < maxSteps && !done.any() {
        let (a, logA) = policy(o1)
        let v = value(o1).squeezingShape(at: 1)
        let (o2, r, d, _) = try env.step(a)
        steps.append(PPOReplayMemory.Step(o: o1, a: a, oldLogP: logA, r: r, v: v))
        o1 = o2
        done = d
        episodeReward += r.sum().scalarized()
        t += 1
    }
    let vals = value(o1).squeezingShape(at: 1)
    let lastVal = (1 - Tensor<Float>(done)) * vals
    memory.append(steps: steps, lastVal: lastVal)
    print(episodeReward/Float(n))
    return t
}

private func improvePolicy(o: Tensor<Float>, a: Tensor<Float>, oldLogP: Tensor<Float>, adv: Tensor<Float>) -> Bool {
    let (logA, backprop) = valueWithPullback(at: policy) { model -> Tensor<Float> in
        return PPOHelpers.logLikelihood(x: a, mu: model(o), logStd: model.logStd).sum(alongAxes: -1)
    }
    let dLoss = gradient(at: logA) { logA in
        return PPOHelpers.policyLoss(oldLogP: oldLogP, logP: logA, adv: adv)
    }
    let dPolicy = backprop(dLoss)
    pOpt.update(&policy, along: dPolicy)
    return PPOHelpers.approxKL(oldLogP: oldLogP, logP: logA) > targetKL
}

private func improveValue(o: Tensor<Float>, r: Tensor<Float>) {
    let dValue = gradient(at: value) { model -> Tensor<Float> in
        let v = model(o)
        return meanSquaredError(predicted: v, expected: r)
    }
    vOpt.update(&value, along: dValue)
}

do {
    let steps = 1000
    for i in 1...30 {
        print("Starting training epoch \(i)")
        memory = PPOReplayMemory(steps: steps, n: n, obsShape: env.observationSpace.shape)
        var j = 0
        while j < steps {
            j += try runEpisode(maxSteps: steps-j)
        }
        let (o, a, oldLogP, r, adv) = memory.getBatch()
        for _ in 0..<25 {
            if improvePolicy(o: o, a: a, oldLogP: oldLogP, adv: adv) { break }
        }
        for _ in 0..<40 {
            improveValue(o: o, r: r)
        }
    }
} catch {
    print(error)
}
