import TensorFlow
import S4TFUnityGym

let path = "/...PATHTO/Basic.app"
guard let env = try? UnityGym(path) else {
    exit(0)
}
defer { env.close() }

let inputLen = env.observationSpace.shape.contiguousSize
let actionDim: Int = env.actionSpace.shape[0]
let maxReward: Float = 0.93

var model = BehaviorModel(inputSize: inputLen, outputSize: actionDim)
let opt = Adam(for: model)
var memory = [Episode]()
var eps: Float = 1.0
Context.local.learningPhase = .training

func action(o: Tensor<Float>, dh: Tensor<Float>, dr: Tensor<Float>) -> Tensor<Int32> {
    if Float.random(in: 0..<1) < eps {
        return env.actionSpace.sample()
    }
    return model(o, dh, dr).argmax()
}

func episode(_ dh: Tensor<Float>, _ dr: Tensor<Float>) throws -> Episode {
    var dh = dh; var dr = dr
    var steps = [Episode.Step]()
    var episodeReward: Float = 0.0
    var o1 = try env.reset()
    for _ in 0..<100 {
        let a = action(o: o1, dh: dh, dr: dr)
        let (o2, r, d, _) = try env.step(a)
        steps.append(Episode.Step(o: o1[0], a: a, r: r))
        dh = max(dh - 1.0, 1.0)
        dr = min(dr - r, maxReward)
        o1 = o2
        episodeReward += r.scalarized()
        if d.scalarized() { break }
    }
    print(episodeReward)
    return Episode(steps: steps, r: episodeReward)
}

do {
    for i in 1...100 {
        print("Epoch \(i)")
        for _ in 1...20 {
            let (dh, dr) = memory.getCommand()
            memory.insertSorted(try episode(dh, dr))
        }
        for _ in 1...200 {
            let (obss, dhs, drs, targets) = memory.sample()
            let dModel = gradient(at: model) { model -> Tensor<Float> in
                let a = model(obss, dhs, drs)
                return softmaxCrossEntropy(logits: a, labels: targets)
            }
            opt.update(&model, along: dModel)
        }
        eps = max(min(0.05, eps * 0.99), 0.01)
    }
} catch {
    print(error)
}
