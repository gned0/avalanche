from avalanche.core import SelfSupervisedPlugin


class MomentumUpdatePlugin(SelfSupervisedPlugin):
    def after_training_iteration(self, strategy, **kwargs):
        strategy.model.momentum_update_all()
