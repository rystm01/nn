
import numpy
import scipy.special


# a three layer neural net
class NeuralNet:

  def __init__(self, input_n, output_n, hidden_n, learning_rate):
    """ num input nodes, 
        num output nodes, 
        num hidden nodes, 
        learning rate"""
    self.input_n = input_n
    self.output_n = output_n
    self.hidden_n = hidden_n
    self.learning_r = learning_rate

    # weights from input to hidden
    self.wh_i = numpy.random.normal(0.0, pow(self.hidden_n, .5),  (self.hidden_n, self.input_n)) - .5

    # weights from hidden to output
    self.wh_o = numpy.random.normal(0.0, pow(self.output_n, .5),  (self.output_n, self.hidden_n)) - .5

    self.activation_function = lambda x : scipy.special.expit(x)



  def train(self, inputs, targets):
    inputs_ = numpy.array(inputs, ndmin=2)
    targets_ = numpy.array(targets, ndmin=2)

    # predict
    hidden_inputs = numpy.dot(self.wh_i, inputs_)
    hidden_outputs = self.activation_function(hidden_inputs)
    final_inputs = numpy.dot(self.wh_o, hidden_outputs)
    preds = self.activation_function(final_inputs)

    # find errors
    out_err = targets_ - preds
    hidden_err = numpy.dot(self.wh_o.T, out_err)
    self.wh_o += self.learning_r * numpy.dot(out_err*preds*(1-preds), numpy.transpose(hidden_outputs))
    self.wh_i += self.learning_r * numpy.dot(hidden_err*hidden_outputs*(1-hidden_outputs) * numpy.transpose(inputs_))



  def predict(self, inputs):
    hidden_inputs = numpy.dot(self.wh_i, inputs)

    hidden_outputs = self.activation_function(hidden_inputs)

    final_inputs = numpy.dot(self.wh_o, hidden_outputs)
    final_outputs = self.activation_function(final_inputs)

    return final_outputs


