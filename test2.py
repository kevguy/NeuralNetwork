#!/usr/bin/python

import sys

import random
import math
import csv

#
# Shorthand:
#   "pd_" as a variable prefix means "partial derivative"
#   "d_" as a variable prefix means "derivative"
#   "_wrt_" is shorthand for "with respect to"
#   "w_ho" and "w_ih" are the index of weights from hidden to output layer neurons and input to hidden layer neurons respectively
#

class NeuralNetwork:
    LEARNING_RATE = 0.9

    def __init__(self, num_inputs, num_hidden, num_kaboom, num_outputs, hidden_layer_weights = None, hidden_layer_bias = None, kaboom_layer_weights = None, kaboom_layer_bias = None, output_layer_weights = None, output_layer_bias = None):
        self.num_inputs = num_inputs

        self.hidden_layer = NeuronLayer(num_hidden, hidden_layer_bias)
        self.output_layer = NeuronLayer(num_outputs, output_layer_bias)
        self.kaboom_layer = NeuronLayer(num_kaboom, kaboom_layer_bias)

        self.init_weights_from_inputs_to_kaboom_layer_neurons(kaboom_layer_weights)
        self.init_weights_from_kaboom_layer_to_hidden_layer_neurons(hidden_layer_weights)
        self.init_weights_from_hidden_layer_neurons_to_output_layer_neurons(output_layer_weights)

    def init_weights_from_inputs_to_kaboom_layer_neurons(self, kaboom_layer_weights):
        weight_num = 0
        for h in range(len(self.kaboom_layer.neurons)):
            for i in range(self.num_inputs):
                if not kaboom_layer_weights:
                    self.kaboom_layer.neurons[h].weights.append(random.random())
                else:
                    self.kaboom_layer.neurons[h].weights.append(kaboom_layer_weights[weight_num])
                weight_num += 1

    def init_weights_from_kaboom_layer_to_hidden_layer_neurons(self, hidden_layer_weights):
        weight_num = 0
        for h in range(len(self.hidden_layer.neurons)):
            for i in range(len(self.kaboom_layer.neurons)):
                if not hidden_layer_weights:
                    self.hidden_layer.neurons[h].weights.append(random.random())
                else:
                    self.hidden_layer.neurons[h].weights.append(hidden_layer_weights[weight_num])
                weight_num += 1

    def init_weights_from_hidden_layer_neurons_to_output_layer_neurons(self, output_layer_weights):
        weight_num = 0
        for o in range(len(self.output_layer.neurons)):
            for h in range(len(self.hidden_layer.neurons)):
                if not output_layer_weights:
                    self.output_layer.neurons[o].weights.append(random.random())
                else:
                    self.output_layer.neurons[o].weights.append(output_layer_weights[weight_num])
                weight_num += 1

    def inspect(self):
        print('------')
        print('* Inputs: {}'.format(self.num_inputs))
        print('------')
        print('Kaboom Layer')
        self.kaboom_layer.inspect()
        print('------')
        print('Hidden Layer')
        self.hidden_layer.inspect()
        print('------')
        print('* Output Layer')
        self.output_layer.inspect()
        print('------')

    def feed_forward(self, inputs):
        kaboom_layer_outputs = self.kaboom_layer.feed_forward(inputs)
        hidden_layer_outputs = self.hidden_layer.feed_forward(kaboom_layer_outputs)
        return self.output_layer.feed_forward(hidden_layer_outputs)

    # Uses online learning, ie updating the weights after each training case
    def train(self, training_inputs, training_outputs):
        self.feed_forward(training_inputs)

        # 1. Output neuron deltas
        pd_errors_wrt_output_neuron_total_net_input = [0] * len(self.output_layer.neurons)
        for o in range(len(self.output_layer.neurons)):

            # dE/dz_j
            pd_errors_wrt_output_neuron_total_net_input[o] = self.output_layer.neurons[o].calculate_pd_error_wrt_total_net_input(training_outputs[o])

        # 2. Hidden neuron deltas
        pd_errors_wrt_hidden_neuron_total_net_input = [0] * len(self.hidden_layer.neurons)
        for h in range(len(self.hidden_layer.neurons)):

            # We need to calculate the derivative of the error with respect to the output of each hidden layer neuron
            d_error_wrt_hidden_neuron_output = 0
            for o in range(len(self.output_layer.neurons)):
                d_error_wrt_hidden_neuron_output += pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].weights[h]

            
            pd_errors_wrt_hidden_neuron_total_net_input[h] = d_error_wrt_hidden_neuron_output * self.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_input()

        pd_errors_wrt_kaboom_neuron_total_net_input = [0] * len(self.kaboom_layer.neurons)
        for h in range(len(self.kaboom_layer.neurons)):
            
            d_error_wrt_kaboom_neuron_output = 0
            for o in range(len(self.hidden_layer.neurons)):
                d_error_wrt_kaboom_neuron_output += pd_errors_wrt_hidden_neuron_total_net_input[o] * self.hidden_layer.neurons[o].weights[h]

            pd_errors_wrt_kaboom_neuron_total_net_input[h] = d_error_wrt_kaboom_neuron_output * self.kaboom_layer.neurons[h].calculate_pd_total_net_input_wrt_input()


        # 3. Update output neuron weights
        for o in range(len(self.output_layer.neurons)):
            for w_ho in range(len(self.output_layer.neurons[o].weights)):

                pd_error_wrt_weight = pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].calculate_pd_total_net_input_wrt_weight(w_ho)

                self.output_layer.neurons[o].weights[w_ho] -= self.LEARNING_RATE * pd_error_wrt_weight

        # 4. Update hidden neuron weights
        for h in range(len(self.hidden_layer.neurons)):
            for w_ih in range(len(self.hidden_layer.neurons[h].weights)):

                pd_error_wrt_weight = pd_errors_wrt_hidden_neuron_total_net_input[h] * self.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_weight(w_ih)

                self.hidden_layer.neurons[h].weights[w_ih] -= self.LEARNING_RATE * pd_error_wrt_weight

        for k in range(len(self.kaboom_layer.neurons)):
            for w_ik in range(len(self.kaboom_layer.neurons[k].weights)):

                pd_error_wrt_weight = pd_errors_wrt_kaboom_neuron_total_net_input[k] * self.kaboom_layer.neurons[k].calculate_pd_total_net_input_wrt_weight(w_ik)

                self.kaboom_layer.neurons[k].weights[w_ik] -= self.LEARNING_RATE * pd_error_wrt_weight

    def checking(self, training_sets):
        total_error = 0
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
       

        count = 0

        text_file = open('checking.txt', "w")

        for t in range(len(training_sets)):
            
            weak = 0
            strong = 0



            

            training_inputs, training_outputs = training_sets[t]

            routput = self.feed_forward(training_inputs)
            #print routput

            text_file.write('sample '+ str(t) + str(training_outputs) + ' ' + str(self.output_layer.neurons[0].output) + str(self.output_layer.neurons[1].output) + '\n')

            #print training_inputs

            if training_outputs[0] < training_outputs[1]:
                weak = 1
            else:
                strong = 1
            
            
            # weak
            if self.output_layer.neurons[0].output >= self.output_layer.neurons[1].output:
                if weak == 1:
                    true_negative = true_negative + 1
                else:
                    false_negative = false_negative + 1
                    print 'wrong false_negative' + str(t)
            else:
                # strong
                if weak == 1:
                    false_positive = false_positive + 1
                    print 'wrong false_positive' + str(t)
                else:
                    true_positive = true_positive + 1

            count = count + 1

        print 'true_positive'
        print true_positive
        print 'true_negative'
        print true_negative
        print 'false_positive'
        print false_positive
        print 'false_negative'
        print false_negative
        print 'set size'
        print len(training_sets)
        print 'accuracy'
        print float(((true_positive + true_negative) / len(training_sets)))

        text_file.close()

        return true_positive, true_negative, false_positive, false_negative






    def calculate_total_error(self, training_sets):
        total_error = 0
        for t in range(len(training_sets)):
            training_inputs, training_outputs = training_sets[t]
            self.feed_forward(training_inputs)
            for o in range(len(training_outputs)):
                total_error += self.output_layer.neurons[o].calculate_error(training_outputs[o])
        return total_error




class NeuronLayer:
    def __init__(self, num_neurons, bias):

        # Every neuron in a layer shares the same bias
        self.bias = bias if bias else random.random()
        #self.bias = 0

        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(Neuron(self.bias))

    def inspect(self):
        print('Neurons:', len(self.neurons))
        for n in range(len(self.neurons)):
            print(' Neuron', n)
            for w in range(len(self.neurons[n].weights)):
                print('  Weight:', self.neurons[n].weights[w])
            print('  Bias:', self.bias)

    def feed_forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs))
        return outputs

    def get_outputs(self):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return outputs

class Neuron:
    def __init__(self, bias):
        self.bias = bias
        self.weights = []

    def routput(self):
        return self.output

    def calculate_output(self, inputs):
        self.inputs = inputs
        self.output = self.squash(self.calculate_total_net_input())
        return self.output

    def calculate_total_net_input(self):
        total = 0
        for i in range(len(self.inputs)):
            total += self.inputs[i] * self.weights[i]
        return total + self.bias

    # Apply the logistic function to squash the output of the neuron
    # The result is sometimes referred to as 'net' [2] or 'net' [1]
    def squash(self, total_net_input):
        return 1 / (1 + math.exp(-total_net_input))

    # Determine how much the neuron's total input has to change to move closer to the expected output
    #
    # Now that we have the partial derivative of the error with respect to the output (dE/dy_j) and
    # the derivative of the output with respect to the total net input (dy_j/dz_j) we can calculate
    # the partial derivative of the error with respect to the total net input.
    #
    def calculate_pd_error_wrt_total_net_input(self, target_output):
        return self.calculate_pd_error_wrt_output(target_output) * self.calculate_pd_total_net_input_wrt_input();

    # The error for each neuron is calculated by the Mean Square Error method:
    def calculate_error(self, target_output):
        return 0.5 * (target_output - self.output) ** 2

    # The partial derivate of the error with respect to actual output then is calculated by:
    # = 2 * 0.5 * (target output - actual output) ^ (2 - 1) * -1
    # = -(target output - actual output)
    #
    # The Wikipedia article on backpropagation [1] simplifies to the following, but most other learning material does not [2]
    # = actual output - target output
    #
    # Alternative, you can use (target - output), but then need to add it during backpropagation [3]
    #
    # Note that the actual output of the output neuron is often written as y_j and target output as t_j so:
    def calculate_pd_error_wrt_output(self, target_output):
        return -(target_output - self.output)

    # The total net input into the neuron is squashed using logistic function to calculate the neuron's output:
    # 
    # Note that where  represents the output of the neurons in whatever layer we're looking at and i represents the layer below it
    #
    # The derivative (not partial derivative since there is only one variable) of the output then is:
    # 
    def calculate_pd_total_net_input_wrt_input(self):
        return self.output * (1 - self.output)

    # The total net input is the weighted sum of all the inputs to the neuron and their respective weights:
    # 
    #
    # The partial derivative of the total net input with respective to a given weight (with everything else held constant) then is:
    # 
    def calculate_pd_total_net_input_wrt_weight(self, index):
        return self.inputs[index]


def readCSV(inputFile, small_sample_mode):
    cr = csv.reader(open(inputFile,"rb"))
    training_sets=[]
    validation_sets=[]
    count = 0

    for row in cr:

        # print row
        
        # print index 2, second to last entry
        # print row[2], row[-2] 
        whole = []
        inputt = []
        output = []
        """
        for i in range(0,46):
            inputt.append(float(row[i]))
        output.append(float(row[46]))
        output.append(float(row[47]))
        """
        for i in range(0,23):
            inputt.append(float(row[i]))
        output.append(float(row[23]))
        output.append(float(row[24]))
        whole.append(inputt)
        whole.append(output)
        #if (count <= 2130):
        training_sets.append(whole)
        """
        if (count <= 2300):
            training_sets.append(whole)
        else:
            validation_sets.append(whole)
        #print len(input)
        """
        count = count + 1
    
    print count

    if (len(training_sets) > 850 and small_sample_mode == 1):
        training_sets = training_sets[start:851]
        validation_sets = validation_sets[start:851]

    return training_sets, validation_sets


def makeNN(count, nn):
    outputFile = 'best' + str(count) + ".nn"

    text_file = open(outputFile, "w")
    #text_file.write("Purchase Amount: %s" % TotalAmount)

            
    output = str(nn.LEARNING_RATE) + '\n'
    text_file.write(output)
    output =  str(nn.num_inputs) + ' ' + str(len(nn.kaboom_layer.neurons)) + ' ' + str(len(nn.hidden_layer.neurons)) + ' ' + str(len(nn.output_layer.neurons))
    text_file.write(output + '\n')
    output = 'I ' + str(nn.num_inputs) + ' ' + 'H ' + str(len(nn.kaboom_layer.neurons))
    text_file.write(output + '\n')

    for i in range(len(nn.kaboom_layer.neurons)):
        output = ''
        for j in range(nn.num_inputs):
                output += str(nn.kaboom_layer.neurons[i].weights[j]) + ' '
        output += str(nn.kaboom_layer.bias)
        text_file.write(output + '\n')

    output = 'H ' + str(len(nn.kaboom_layer.neurons)) + ' ' + 'H ' + str(len(nn.hidden_layer.neurons))
    text_file.write(output + '\n')
    for i in range(len(nn.hidden_layer.neurons)):
        output = ''
        for j in range(len(nn.kaboom_layer.neurons)):
                output += str(nn.hidden_layer.neurons[i].weights[j]) + ' '
        output += str(nn.hidden_layer.bias)
        text_file.write(output + '\n')

    output = 'H ' + str(len(nn.hidden_layer.neurons)) + ' ' + 'O ' + str(len(nn.output_layer.neurons))
    text_file.write(output + '\n')
    for i in range(len(nn.output_layer.neurons)):
        output = ''
        for j in range(len(nn.hidden_layer.neurons)):
                output += str(nn.output_layer.neurons[i].weights[j]) + ' '
        output += str(nn.output_layer.bias)
        text_file.write(output + '\n')
            
    text_file.close()







if __name__ == "__main__":
    inputFile = sys.argv[1]
    answerFile = sys.argv[2]
    outputFile = sys.argv[3]
    maxIter = sys.argv[4]
    small_sample_mode = sys.argv[5]


    #kaboom_wig = [0.3780176374,0.515331894047,0.291106914321,0.320622689812,0.883800041234,0.825343523357,0.630509682184,1.00605247895,0.784249274122,0.753895199231,0.209683206687,0.918856053356,0.913350411701,0.704052298663,0.788489728513,0.394125872922,0.929183952011,0.424444081346,0.655429257637,0.281422142755,0.173735132142,0.62513988455,0.598743697395,-1.6333372382,-0.8615359486,-1.3287733076,0.0557187893,0.1847816445,0.2131464236,-0.3626520925,-0.004712857,-0.5550694605,0.2345280707,0.5734106134,0.0796327303,-0.1999143981,-0.1192528943,-0.167520301,-0.3682821165,-0.0687812868,0.1152530174,-0.4864253706,-0.0481637346,0.128777975,-0.330954912,5.984176299,-0.3853127975,-0.3467797068,-0.9585396549,0.8244140229,0.1850419376,0.1002687359,0.2946441697,-0.0617442495,0.3803301491,-0.2486745539,0.1410288623,0.5814831181,0.0491362509,0.3154042562,0.595185145,-0.1575723533,-0.0260724687,0.72803044,0.730444195,0.0984334264,0.3372292691,0.3045519641,1.647247103,0.3898970848,0.6603391909,0.3537721492,0.1007954389,0.8372797487,0.2043380776,0.6591563092,0.5998250917,0.3306927004,0.2448881123,0.4583250012,0.1387395568,0.6726465499,0.9653258016,0.0490061745,0.4140524284,0.4128670307,0.9801123075,0.7984096951,0.7009440245,0.3398546893,0.1765338865,0.3950363201,-0.1905507287,0.3844567164,0.3031256866,0.0692906875,0.105299992,0.2152496514,0.4451250236,0.0629255386,0.8185533258,0.3335191619,0.0401508314,0.5469796393,0.5807640067,0.5762102354,0.4908353542,0.748294984,0.3595553038,0.41431696,0.18291389,0.915693214,0.6537213041,0.7336867125,0.744822221]

    #hidden_wig = [0.7086352562,-3.7369564413,-0.0692063438,0.7760746482,0.1126856698,0.4873038923,-2.8286360735,-0.2786346271,0.6451781524,0.042551669,-1.0259113047,4.3426663038,0.9190067835,-0.914876525,-0.6246641758,-0.9528374409,3.9670396386,0.6588225704,-0.4186176047,-0.8091249714,0.0878644241,-1.6936131653,-0.4822735002,0.6191988847,-0.2884850053,0.0403801701,-1.4748485466,-0.1170703635,-0.0498742521,-0.2275508238,0.5488861102,-2.6480279645,0.2001726507,0.1776936801,-0.0352792847,-0.1221553214,-0.4859673442,-0.8420959914,-0.4442511822,-0.3528500364,-0.2277621464,-0.8999886654,-0.7415785147,-0.112198019,-0.0700384119,-0.0381198309,0.2757310723,0.3076868618,-0.5970641802,0.025855127]

    #output_wig = [-2.7953320549, -1.96765278863,3.9955969839,3.3768357072,-1.6740836228,-0.9508453559,-2.1648106669,-0.4319631818,-0.9325685805,0.5257452777, 3.0183129164, 2.37643608964,-3.8285177639, -3.65647289146,0.9159828908,1.3672326956,1.7057748593,0.5903710623,0.6263652097,-0.1918076993]



    training_sets, validation_sets = readCSV(inputFile, small_sample_mode)
    print training_sets[0]
    print training_sets[99]
    print len(training_sets[0][0])
    print len(training_sets[0][1])

    #nn = NeuralNetwork(len(training_sets[0][0]), 10, 5, len(training_sets[0][1]), kaboom_layer_bias=0, hidden_layer_bias=0, output_layer_bias=0, kaboom_layer_weights = kaboom_wig, hidden_layer_weights = hidden_wig, output_layer_weights = output_wig)
    nn = NeuralNetwork(len(training_sets[0][0]), 10, 10, len(training_sets[0][1]), kaboom_layer_bias=0.1, hidden_layer_bias=0.1, output_layer_bias=0.1   )
    for i in range(int(maxIter)):
        training_inputs, training_outputs = random.choice(training_sets)
        nn.train(training_inputs, training_outputs)
        print(i, nn.calculate_total_error(training_sets))
        makeNN(i, nn)

    routput = nn.feed_forward(training_sets[99][0])
    print routput 
    routput = nn.feed_forward(training_sets[5][0])
    print routput
    nn.checking(training_sets)


    print nn.LEARNING_RATE
    print str(nn.num_inputs) + ' ' + str(len(nn.kaboom_layer.neurons)) + ' ' + str(len(nn.hidden_layer.neurons)) + ' ' + str(len(nn.output_layer.neurons))
    print 'I ' + str(nn.num_inputs) + ' ' + 'H ' + str(len(nn.kaboom_layer.neurons))

    for i in range(nn.num_inputs):
        output = ''
        for j in range(len(nn.kaboom_layer.neurons)):
                output += str(nn.kaboom_layer.neurons[j].weights[i]) + ' '
        output += str(nn.kaboom_layer.bias)
        print output

    print 'H ' + str(len(nn.kaboom_layer.neurons)) + ' ' + 'H ' + str(len(nn.hidden_layer.neurons))

    for i in range(len(nn.kaboom_layer.neurons)):
        output = ''
        for j in range(len(nn.hidden_layer.neurons)):
                output += str(nn.hidden_layer.neurons[j].weights[i]) + ' '
        output += str(nn.hidden_layer.bias)
        print output

    print 'H ' + str(len(nn.hidden_layer.neurons)) + ' ' + 'O ' + str(len(nn.output_layer.neurons))

    for i in range(len(nn.hidden_layer.neurons)):
        output = ''
        for j in range(len(nn.output_layer.neurons)):
                output += str(nn.output_layer.neurons[j].weights[i]) + ' '
        output += str(nn.output_layer.bias)
        print output
            

    text_file = open(outputFile, "w")
    #text_file.write("Purchase Amount: %s" % TotalAmount)

            
    output = str(nn.LEARNING_RATE) + '\n'
    text_file.write(output)
    output =  str(nn.num_inputs) + ' ' + str(len(nn.kaboom_layer.neurons)) + ' ' + str(len(nn.hidden_layer.neurons)) + ' ' + str(len(nn.output_layer.neurons))
    text_file.write(output + '\n')
    output = 'I ' + str(nn.num_inputs) + ' ' + 'H ' + str(len(nn.kaboom_layer.neurons))
    text_file.write(output + '\n')

    for i in range(len(nn.kaboom_layer.neurons)):
        output = ''
        for j in range(nn.num_inputs):
                output += str(nn.kaboom_layer.neurons[i].weights[j]) + ' '
        output += str(nn.kaboom_layer.bias)
        text_file.write(output + '\n')

    output = 'H ' + str(len(nn.kaboom_layer.neurons)) + ' ' + 'H ' + str(len(nn.hidden_layer.neurons))
    text_file.write(output + '\n')
    for i in range(len(nn.hidden_layer.neurons)):
        output = ''
        for j in range(len(nn.kaboom_layer.neurons)):
                output += str(nn.hidden_layer.neurons[i].weights[j]) + ' '
        output += str(nn.hidden_layer.bias)
        text_file.write(output + '\n')

    output = 'H ' + str(len(nn.hidden_layer.neurons)) + ' ' + 'O ' + str(len(nn.output_layer.neurons))
    text_file.write(output + '\n')
    for i in range(len(nn.output_layer.neurons)):
        output = ''
        for j in range(len(nn.hidden_layer.neurons)):
                output += str(nn.output_layer.neurons[i].weights[j]) + ' '
        output += str(nn.output_layer.bias)
        text_file.write(output + '\n')
            
    text_file.close()