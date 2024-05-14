

class NaryExample(object):
    """
    A single training/test example of H-Fact.
    """

    def __init__(self,
                 arity,
                 node_num,
                 hyperedge_num,
                 relation,
                 head,
                 tail,
                 auxiliary_info=None):

        self.arity = arity
        self.node_num = node_num
        self.hyperedge_num = hyperedge_num
        self.relation = relation
        self.head = head
        self.tail = tail
        self.auxiliary_info = auxiliary_info