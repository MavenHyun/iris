import iris_model as iris

def run_iris():
    dataset = iris.iris_bank(90, 30)
    maven = iris.iris_seeker(dataset, 0.88)
    input0 = maven.input_layer()
    hidden0 = maven.single_hidden_layer(input0, 100)
    output0 = maven.output_layer(hidden0, 100)
    maven.optimize(output0, 10000, 0.001)
