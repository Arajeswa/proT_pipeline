class Sequencer():
    def __init__(
        self,
        df = None,
        features    :  list = [],
        id_label    :  str = "id",
        sort_label  : str = "",
        max_seq_len : int = 1500
    ):
        self.df = df
        self.features = features
        self.id_label = id_label
        self.sort_label = sort_label
        self.max_seq_len = max_seq_len
        
    def get_ids(self):
        return self.df[self.id_label].unique().tolist()

    def get_seq(self,id):
        
        df = self.df
        features = self.features
        id_label = self.id_label
        sort_label = self.sort_label
        max_seq_len = self.max_seq_len
        
        arr_h = []
        
        
        for j in features:
            fea = df.set_index(id_label).loc[id].sort_values(sort_label)[j].tolist()
            n_zeros = max_seq_len-len(fea)
            if n_zeros < 0:
                raise ValueError(f"Choose a maximum sequence length > {len(fea)}, id {id} overshoots!")
            fea.extend([0 for _ in range(n_zeros)])
            arr_h.append(fea)

        return arr_h
