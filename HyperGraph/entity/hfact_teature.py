class NaryFeature(object):


    def __init__(self,
                 feature_id,
                 example_id,
                 input_tokens,
                 input_ids,
                 input_mask,
                 mask_position,
                 mask_label,
                 mask_type,
                 arity,
                 incidence_matrix_T,
                 node_num,
                 hyperedge_num,
                 mask_edge_index,
                 node_type,
                 g_input_ids, g_incidence_matrix, g_node_type):

        self.feature_id = feature_id
        self.example_id = example_id
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.mask_position = mask_position
        self.mask_label = mask_label
        self.mask_type = mask_type
        self.arity = arity
        self.incidence_matrix_T = incidence_matrix_T
        self.node_num = node_num
        self.hyperedge_num = hyperedge_num
        self.mask_edge_index = mask_edge_index
        self.node_type = node_type
        self.g_input_ids = g_input_ids
        self.g_incidence_matrix = g_incidence_matrix
        self.g_node_type = g_node_type
