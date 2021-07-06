import tensorflow.compat.v1 as tf
import tensorflow_compression as tfc


class AnalysisTransform(tf.keras.layers.Layer):
    """The analysis transform."""

    def __init__(self, num_filters, *args, **kwargs):
        self.num_filters = num_filters
        super(AnalysisTransform, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self._layers = [
            tfc.SignalConv3D(
                self.num_filters, (5, 5, 5), name="layer_0", corr=True, strides_down=2,
                padding="same_reflect", use_bias=True,
                activation=tf.sigmoid),
            tfc.SignalConv3D(
                self.num_filters, (5, 5, 5), name="layer_1", corr=True, strides_down=2,
                padding="same_reflect", use_bias=True,
                activation=tf.sigmoid),
            tfc.SignalConv3D(
                self.num_filters, (5, 5, 5), name="layer_2", corr=True, strides_down=2,
                padding="same_reflect", use_bias=False,
                activation=None),
        ]
        super(AnalysisTransform, self).build(input_shape)

    def call(self, tensor):
        for layer in self._layers:
            tensor = layer(tensor)
        return tensor


class SynthesisTransform(tf.keras.layers.Layer):
    """The synthesis transform."""

    def __init__(self, num_filters, *args, **kwargs):
        self.num_filters = num_filters
        super(SynthesisTransform, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self._layers = [
            tfc.SignalConv3D(
                self.num_filters, (5, 5, 5), name="layer_0", corr=False, strides_up=2,
                padding="same_reflect", use_bias=True,
                activation=tf.sigmoid),
            tfc.SignalConv3D(
                self.num_filters, (5, 5, 5), name="layer_1", corr=False, strides_up=2,
                padding="same_reflect", use_bias=True,
                activation=tf.sigmoid),
            tfc.SignalConv3D(
                1, (5, 5, 5), name="layer_2", corr=False, strides_up=2,
                padding="same_reflect", use_bias=True,
                activation=tf.sigmoid),
        ]
        super(SynthesisTransform, self).build(input_shape)

    def call(self, tensor):
        for layer in self._layers:
            tensor = layer(tensor)
        return tensor


class HyperAnalysisTransform(tf.keras.layers.Layer):
    """The analysis transform for the entropy model parameters."""

    def __init__(self, num_filters, *args, **kwargs):
        self.num_filters = num_filters
        super(HyperAnalysisTransform, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self._layers = [
            tfc.SignalConv3D(
                self.num_filters, (3, 3, 3), name="layer_0", corr=True, strides_down=1,
                padding="same_zeros", use_bias=True,
                activation=tf.nn.relu),
            tfc.SignalConv3D(
                self.num_filters, (3, 3, 3), name="layer_1", corr=True, strides_down=2,
                padding="same_zeros", use_bias=True,
                activation=tf.nn.relu),
            tfc.SignalConv3D(
                self.num_filters, (3, 3, 3), name="layer_2", corr=True, strides_down=2,
                padding="same_zeros", use_bias=False,
                activation=None),
        ]
        super(HyperAnalysisTransform, self).build(input_shape)

    def call(self, tensor):
        for layer in self._layers:
            tensor = layer(tensor)
        return tensor


class HyperSynthesisTransform(tf.keras.layers.Layer):
    """The synthesis transform for the entropy model parameters."""

    def __init__(self, num_filters, *args, **kwargs):
        self.num_filters = num_filters
        super(HyperSynthesisTransform, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self._layers = [
            tfc.SignalConv3D(
                self.num_filters, (3, 3, 3), name="layer_0", corr=False, strides_up=2,
                padding="same_zeros", use_bias=True, kernel_parameterizer=None,
                activation=tf.nn.relu),
            tfc.SignalConv3D(
                self.num_filters, (3, 3, 3), name="layer_1", corr=False, strides_up=2,
                padding="same_zeros", use_bias=True, kernel_parameterizer=None,
                activation=tf.nn.relu),
            tfc.SignalConv3D(
                self.num_filters, (3, 3, 3), name="layer_2", corr=False, strides_up=1,
                padding="same_zeros", use_bias=True, kernel_parameterizer=None,
                activation=None),
        ]
        super(HyperSynthesisTransform, self).build(input_shape)

    def call(self, tensor):
        for layer in self._layers:
            tensor = layer(tensor)
        return tensor
