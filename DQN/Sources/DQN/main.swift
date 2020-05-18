import TensorFlow
import S4TFUnityGym

let path = "/...PATHTO/Basic.app"
guard let env = try? UnityGym(path) else {
    exit(0)
}
defer { env.close() }

let inputLen = env.observationSpace.shape.contiguousSize
let actionDim: Int = env.actionSpace.shape[0]

var model = Sequential {
    Dense<Float>(inputSize: inputLen, outputSize: 64, activation: relu)
    Dense<Float>(inputSize: 64, outputSize: 64, activation: relu)
    Dense<Float>(inputSize: 64, outputSize: actionDim)
}
var target = model
var opt = Adam(for: model)
let gamma: Float = 0.99
var eps: Float = 0.07
Context.local.learningPhase = .training

func action(o: Tensor<Float>) -> Tensor<Int32> {
    if Float.random(in: 0..<1) < eps {
        return env.actionSpace.sample(squeezingShape: false)
    }
    return model(o).argmax(squeezingAxis: 1)
}

func runEpisodes(_ i: Int) throws -> [DQNEpisode] {
    var episodes = [DQNEpisode]()
    for _ in 0..<i {
        var steps = [DQNEpisode.Step]()
        var episodeReward: Float = 0.0
        var o1 = try env.reset()
        for _ in 0..<100 {
            let a = action(o: o1)
            let (o2, r, d, _) = try env.step(a)
            steps.append(DQNEpisode.Step(o1: o1[0], o2: o2[0], a: a, r: r, d: d))
            o1 = o2
            episodeReward += r.scalarized()
            if d.scalarized() { break }
        }
        episodes.append(DQNEpisode(steps: steps))
        print(episodeReward)
    }
    return episodes
}

func improveModels(_ episodes: [DQNEpisode], i: Int) {
    for _ in 0..<i {
        for episode in episodes {
            let (o1, o2, a, r, d) = episode.getBatch()
            let dModel = gradient(at: model) { model -> Tensor<Float> in
                let Qmax = target(o2).max(alongAxes: 1)
                let Qtar = r + (1 - d) * gamma * Qmax
                let Q = model(o1).batchGathering(atIndices: a, alongAxis: 1)
                return meanSquaredError(predicted: Q, expected: Qtar)
            }
            opt.update(&model, along: dModel)
            target.softUpdate(from: model)
        }
    }
}

do {
    for i in 1...200 {
        print("Starting training epoch \(i)")
        let episodes = try runEpisodes(20)
        improveModels(episodes, i: 3)
        eps = max(0.01, eps * 0.99)
    }
} catch {
    print(error)
}
