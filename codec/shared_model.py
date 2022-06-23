class SharedModel:
    model_weight_list: list = []
    local_epoch = 3
    turn = 1
    one_epoch_batches = 9
    @staticmethod
    def get_model_weight_list():
        """当前worker接收完成之后应该有：
            len = len(trainable_var)
            model_weight_list[i].shape = (node_count, var.shape),
            eg1: (7, 26, 80)
            eg2: (7, 4, 20)
            eg3: (7, 6, 21)
            eg4: (7, 8, 41)

        """
        return SharedModel.model_weight_list

    @staticmethod
    def get_turn():
        """每隔多少个本地batch，进行一轮全局通信"""
        return SharedModel.turn

    @staticmethod
    def get_local_epoch():
        """每隔多少个本地epoch，进行一轮全局通信"""
        return SharedModel.local_epoch

    @staticmethod
    def get_batches():
        """每隔多少个本地epoch，进行一轮全局通信"""
        return SharedModel.one_epoch_batches
