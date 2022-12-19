"""
CFG dumped from the following code

    int loops(int a, int b) {
        int result = 0;

        if (a == b)
            return 0;

        for (int i = 0 ; i < a ; i++) {
            if (i % 2) {
                result += i;
            } else {
                result <<= i;
            }
            result += b;
            for (int j = i ; j < b ; j++) {
                if (j % 4)
                    result += 1;

                result *= j;
            }
        }

        return result;
    }

"""
cfg_loops = {
    1048576: {
        'ins' : [], 'outs': [1048608, 1048601],
        'code': ['local_c = COPY const_0',
                 'register_206 = INT_EQUAL param_1, param_2',
                 'CBRANCH ram_100020, register_206']},
    1048608: {
        'ins': [1048576], 'outs': [1048697], 'code': ['local_10 = COPY const_0', 'BRANCH ram_100079']},
    1048697: {
        'ins' : [1048608, 1048693], 'outs': [1048708, 1048617],
        'code': ['local_c = MULTIEQUAL local_c, local_c', 'local_10 = MULTIEQUAL local_10, local_10',
                 'unique_1000001d = CAST local_10', 'unique_ce80 = INT_SLESS unique_1000001d, param_1',
                 'CBRANCH ram_100029, unique_ce80']},
    1048708: {
        'ins' : [1048601, 1048697], 'outs': [],
        'code': ['local_c = MULTIEQUAL local_c, local_c', 'RETURN const_0, local_c']},
    1048617: {
        'ins' : [1048697], 'outs': [1048627, 1048635],
        'code': ['register_0 = INT_AND local_10, const_1', 'register_206 = INT_EQUAL register_0, const_0',
                 'CBRANCH ram_10003b, register_206']},
    1048627: {
        'ins': [1048617], 'outs': [1048643], 'code': ['local_c = INT_ADD local_c, local_10', 'BRANCH ram_100043']},
    1048643: {
        'ins' : [1048627, 1048635], 'outs': [1048685],
        'code': ['local_c = MULTIEQUAL local_c, local_c', 'local_c = INT_ADD local_c, param_2',
                 'local_14 = COPY local_10',
                 'BRANCH ram_10006d']},
    1048685: {
        'ins' : [1048643, 1048671], 'outs': [1048693, 1048657],
        'code': ['local_c = MULTIEQUAL local_c, local_c', 'local_14 = MULTIEQUAL local_14, local_14',
                 'unique_10000021 = CAST local_14', 'unique_ce80 = INT_SLESS unique_10000021, param_2',
                 'CBRANCH ram_100051, unique_ce80']},
    1048693: {
        'ins': [1048685], 'outs': [1048697], 'code': ['local_10 = INT_ADD local_10, const_1']},
    1048657: {
        'ins' : [1048685], 'outs': [1048671, 1048667],
        'code': ['register_0 = INT_AND local_14, const_3', 'register_206 = INT_NOTEQUAL register_0, const_0',
                 'CBRANCH ram_10005f, register_206']},
    1048671: {
        'ins' : [1048657, 1048667], 'outs': [1048685],
        'code': ['local_c = MULTIEQUAL local_c, local_c', 'local_c = INT_MULT local_c, local_14',
                 'local_14 = INT_ADD local_14, const_1']},
    1048667: {
        'ins': [1048657], 'outs': [1048671], 'code': ['local_c = INT_ADD local_c, const_1']},
    1048635: {
        'ins' : [1048617], 'outs': [1048643],
        'code': ['register_8 = SUBPIECE local_10, const_0', 'unique_55000 = INT_AND register_8, const_1f',
                 'local_c = INT_LEFT local_c, unique_55000']},
    1048601: {
        'ins': [1048576], 'outs': [1048708], 'code': ['local_c = COPY const_0', 'BRANCH ram_100084']}}



"""
    int banzai(int a) {
        if (a > 0) {
            a++;
    
            if (a > 5) {
                a -= 3;
            } else {
                a /= 2;
                goto bazinga;
            }
            a += 32;
        }
        a++;
    bazinga:
        return a;
    }
"""
cfg_if_w_goto = {
    1048840: {
        'ins' : [], 'outs': [1048890, 1048853],
        'code': ['unique_d100 = INT_SLESS const_0, param_1',
                 'local_c = COPY param_1', 'CBRANCH ram_10013a, unique_d100']},
    1048890: {
        'ins' : [1048840, 1048863], 'outs': [],
        'code': ['local_c = MULTIEQUAL local_c, local_c',
                 'unique_bf80 = INT_ADD local_c, const_1',
                 'register_0 = COPY unique_bf80',
                 'RETURN const_0, register_0']},
    1048853: {
        'ins' : [1048840], 'outs': [1048863, 1048873],
        'code': ['unique_bf80 = INT_ADD param_1, const_1',
                 'unique_d100 = INT_SLESS unique_bf80, const_6',
                 'CBRANCH ram_100129, unique_d100']},
    1048863: {
        'ins' : [1048853], 'outs': [1048890],
        'code': ['local_c = INT_ADD param_1, const_1e', 'BRANCH ram_10013a']},
    1048873: {
        'ins' : [1048853], 'outs': [],
        'code': ['register_0 = INT_SDIV unique_bf80, const_2',
                 'register_0 = COPY register_0',
                 'RETURN const_0, register_0']
    }
}

"""
    int goto_you_said(int a) {
        if (a > 0) {
            a *= 2;
            goto l2;
        }
    
        if (a < 0) {
            a /= 2;
            goto l1;
        }
    
        a /= 5;
    
    l2:
        a += 16;
    
    l1:
        a -= 9;
    
        return a;
    }
"""
cfg_goto_you_said = {
    1048899: {
        'ins' : [], 'outs': [1048912, 1048917],
        'code': ['unique_d100 = INT_SLESS param_1, const_1', 'CBRANCH ram_100155, unique_d100']},
    1048912: {
        'ins' : [1048899], 'outs': [1048969],
        'code': ['local_c = INT_LEFT param_1, const_1', 'BRANCH ram_100189']},
    1048969: {
        'ins' : [1048912, 1048940], 'outs': [1048973],
        'code': ['local_c = MULTIEQUAL local_c, local_c', 'local_c = INT_ADD local_c, const_10']},
    1048973: {
        'ins' : [1048969, 1048923], 'outs': [],
        'code': [
            'local_c = MULTIEQUAL local_c, local_c',
            'unique_bf80 = INT_ADD local_c, const_fffffff7',
            'register_0 = COPY unique_bf80',
            'RETURN const_0, register_0']},
    1048917: {
        'ins' : [1048899], 'outs': [1048940, 1048923],

        'code': ['register_207 = INT_SLESS param_1, const_0', 'CBRANCH ram_10016c, register_207']},
    1048940: {
        'ins' : [1048917], 'outs': [1048969],
        'code': ['local_c = INT_SDIV param_1, const_5']},
    1048923: {
        'ins' : [1048917], 'outs': [1048973],
        'code': ['local_c = INT_SDIV param_1, const_2', 'BRANCH ram_10018d']},
}

cfg_struct = {
    1048982: {
        'ins' : [],
        'outs': [1049107],
        'code': [
            'unique_10000083 = CALL ram_101008, const_3c',
            'miaos = CAST unique_10000083',
            'index = COPY const_0',
            'BRANCH ram_100213'
        ]
    },
    1049107: {
        'ins' : [1048982, 1049095],
        'outs': [1049113, 1049013],
        'code': [
            'index = MULTIEQUAL index, index',
            'unique_d100 = INT_SLESS index, const_5',
            'CBRANCH ram_1001b5, unique_d100'
        ]
    },
    1049113: {
        'ins' : [1049107],
        'outs': [],
        'code': [
            'register_0 = COPY miaos',
            'RETURN const_0, register_0'
        ]
    },
    1049013: {
        'ins' : [1049107],
        'outs': [1049089],
        'code': [
            'register_10 = INT_SEXT index',
            'tmp = PTRADD miaos, register_10',
            'unique_10000043 = PTRSUB tmp, const_0',
            'STORE const_1b1, unique_10000043',
            'count = COPY const_0',
            'BRANCH ram_100201'
        ]
    },
    1049089: {
        'ins' : [1049013, 1049064],
        'outs': [1049095, 1049064],
        'code': [
            'count = MULTIEQUAL count, count',
            'unique_d100 = INT_SLESS count, const_5',
            'CBRANCH ram_1001e8, unique_d100'
        ]
    },
    1049095: {
        'ins' : [1049089],
        'outs': [1049107],
        'code': [
            'unique_10000073 = PTRSUB tmp, const_4',
            'unique_3100 = PTRADD unique_10000073, const_5',
            'STORE const_1b1, unique_3100',
            'index = INT_ADD index, const_1'
        ]
    },
    1049064: {
        'ins' : [1049089],
        'outs': [1049089],
        'code': [
            'unique_10000039 = SUBPIECE index, const_0',
            'register_8 = INT_ADD unique_10000039, const_30',
            'register_0 = INT_SEXT count',
            'unique_1000005b = PTRSUB tmp, const_4',
            'unique_3a00 = PTRADD unique_1000005b, register_0',
            'STORE const_1b1, unique_3a00',
            'count = INT_ADD count, const_1'
        ]
    }
}

"""

    void call_me_w_a_struct(void) {
      struct_miao bau;
      
      bau.index = 0;
      bau.prompt._0_4_ = 0x6162656b;
      bau.prompt._4_2_ = 0x62;
      do_something_w_struct(&bau,3);
      return;
    }
"""

cfg_struct_bis = {
    1049119: {
        'ins' : [],
        'outs': [],
        'code': [
            'stack_-14 = COPY const_0',
            'stack_-10 = COPY const_6162656b',
            'stack_-c = COPY const_62',
            # >>> currentProgram.getRegister(op.getInput(1).getDef().getInput(0))
            # RSP
            'unique_3100 = PTRSUB register_20, const_-14',
            # the second argument to INDIRECT is the Pcode with indirect effects
            # see for example Funcdata::newIndirectOp()
            'stack_-14 = INDIRECT stack_-14, const_1c',
            'stack_-10 = INDIRECT stack_-10, const_1c',
            'stack_-c = INDIRECT stack_-c, const_1c',
            'CALL ram_101010, unique_3100',
            'RETURN const_0'
        ]
    }
}

"""
    void encoding(char* msg, char* encoded) {
        while (*msg) {
            if (*msg == '\r')
                break;
            if (*msg == '\n') {
                *encoded++ = '<';
                *encoded++ = 'b';
                *encoded++ = 'r';
                *encoded++ = '>';
            } else {
                *encoded++ = *msg;
            }
            msg++;
        }
        *encoded = '\0';
    }
"""
func_encoding = {
    'cfg'  : {
        1048692: {
            'code': [('ASSIGNMENT',
                      ('Variable',
                       'local_18',
                       ('Type',
                        'char *',
                        8,
                        [],
                        ('Type', 'char', 1, [], 'None'))),
                      ('Variable',
                       'param_2',
                       ('Type',
                        'char *',
                        8,
                        [],
                        ('Type', 'char', 1, [], 'None')))),
                     ('ASSIGNMENT',
                      ('Variable',
                       'local_10',
                       ('Type',
                        'char *',
                        8,
                        [],
                        ('Type', 'char', 1, [], 'None'))),
                      ('Variable',
                       'param_1',
                       ('Type',
                        'char *',
                        8,
                        [],
                        ('Type', 'char', 1, [], 'None'))))],
            'ins' : [],
            'outs': [1048816]},
        1048706: {'code': [('ControlFlowExpression',
                            ('NOT_EQUAL',
                             ('DEREF',
                              ('Variable',
                               'local_10',
                               ('Type',
                                'char *',
                                8,
                                [],
                                ('Type', 'char', 1, [], 'None')))),
                             ('Constant',
                              13,
                              ('Type', 'char', 1, [], 'None'))))],
                  'ins' : [1048816],
                  'outs': [1048830, 1048717]},
        1048717: {'code': [('ControlFlowExpression',
                            ('EQUAL',
                             ('DEREF',
                              ('Variable',
                               'local_10',
                               ('Type',
                                'char *',
                                8,
                                [],
                                ('Type', 'char', 1, [], 'None')))),
                             ('Constant',
                              10,
                              ('Type', 'char', 1, [], 'None'))))],
                  'ins' : [1048706],
                  'outs': [1048790, 1048728]},
        1048728: {'code': [('ASSIGNMENT',
                            ('DEREF',
                             ('Variable',
                              'local_18',
                              ('Type',
                               'char *',
                               8,
                               [],
                               ('Type', 'char', 1, [], 'None')))),
                            ('Constant', 60, ('Type', 'char', 1, [], 'None'))),
                           ('ASSIGNMENT',
                            ('DEREF',
                             ('ADD',
                              ('Variable',
                               'local_18',
                               ('Type',
                                'char *',
                                8,
                                [],
                                ('Type', 'char', 1, [], 'None'))),
                              ('Constant',
                               1,
                               ('Type', 'long', 8, [], 'None')))),
                            ('Constant', 98, ('Type', 'char', 1, [], 'None'))),
                           ('ASSIGNMENT',
                            ('DEREF',
                             ('ADD',
                              ('Variable',
                               'local_18',
                               ('Type',
                                'char *',
                                8,
                                [],
                                ('Type', 'char', 1, [], 'None'))),
                              ('Constant',
                               2,
                               ('Type', 'long', 8, [], 'None')))),
                            ('Constant',
                             114,
                             ('Type', 'char', 1, [], 'None'))),
                           ('ASSIGNMENT',
                            ('DEREF',
                             ('ADD',
                              ('Variable',
                               'local_18',
                               ('Type',
                                'char *',
                                8,
                                [],
                                ('Type', 'char', 1, [], 'None'))),
                              ('Constant',
                               3,
                               ('Type', 'long', 8, [], 'None')))),
                            ('Constant', 62, ('Type', 'char', 1, [], 'None'))),
                           ('ASSIGNMENT',
                            ('Variable',
                             'local_18',
                             ('Type',
                              'char *',
                              8,
                              [],
                              ('Type', 'char', 1, [], 'None'))),
                            ('ADD',
                             ('Variable',
                              'local_18',
                              ('Type',
                               'char *',
                               8,
                               [],
                               ('Type', 'char', 1, [], 'None'))),
                             ('Constant',
                              4,
                              ('Type', 'long', 8, [], 'None'))))],
                  'ins' : [1048717],
                  'outs': [1048811]},
        1048790: {'code': [('ASSIGNMENT',
                            ('DEREF',
                             ('Variable',
                              'local_18',
                              ('Type',
                               'char *',
                               8,
                               [],
                               ('Type', 'char', 1, [], 'None')))),
                            ('DEREF',
                             ('Variable',
                              'local_10',
                              ('Type',
                               'char *',
                               8,
                               [],
                               ('Type', 'char', 1, [], 'None'))))),
                           ('ASSIGNMENT',
                            ('Variable',
                             'local_18',
                             ('Type',
                              'char *',
                              8,
                              [],
                              ('Type', 'char', 1, [], 'None'))),
                            ('ADD',
                             ('Variable',
                              'local_18',
                              ('Type',
                               'char *',
                               8,
                               [],
                               ('Type', 'char', 1, [], 'None'))),
                             ('Constant',
                              1,
                              ('Type', 'long', 8, [], 'None'))))],
                  'ins' : [1048717],
                  'outs': [1048811]},
        1048811: {'code': [('ASSIGNMENT',
                            ('Variable',
                             'local_10',
                             ('Type',
                              'char *',
                              8,
                              [],
                              ('Type', 'char', 1, [], 'None'))),
                            ('ADD',
                             ('Variable',
                              'local_10',
                              ('Type',
                               'char *',
                               8,
                               [],
                               ('Type', 'char', 1, [], 'None'))),
                             ('Constant',
                              1,
                              ('Type', 'long', 8, [], 'None'))))],
                  'ins' : [1048728, 1048790],
                  'outs': [1048816]},
        1048816: {'code': [('ControlFlowExpression',
                            ('NOT_EQUAL',
                             ('DEREF',
                              ('Variable',
                               'local_10',
                               ('Type',
                                'char *',
                                8,
                                [],
                                ('Type', 'char', 1, [], 'None')))),
                             ('Constant',
                              0,
                              ('Type', 'char', 1, [], 'None'))))],
                  'ins' : [1048692, 1048811],
                  'outs': [1048830, 1048706]},
        1048830: {'code': [('ASSIGNMENT',
                            ('DEREF',
                             ('Variable',
                              'local_18',
                              ('Type',
                               'char *',
                               8,
                               [],
                               ('Type', 'char', 1, [], 'None')))),
                            ('Constant', 0, ('Type', 'char', 1, [], 'None'))),
                           ('RETURN',
                            ('Constant',
                             0,
                             ('Type', 'undefined8', 8, [], 'None')))],
                  'ins' : [1048706, 1048816],
                  'outs': []}},
    'start': 1048692,
    'vars' : [('Variable',
               'param_2',
               ('Type', 'char *', 8, [], ('Type', 'char', 1, [], 'None'))),
              ('Variable',
               'local_18',
               ('Type', 'char *', 8, [], ('Type', 'char', 1, [], 'None'))),
              ('Variable',
               'param_1',
               ('Type', 'char *', 8, [], ('Type', 'char', 1, [], 'None'))),
              ('Variable',
               'local_10',
               ('Type', 'char *', 8, [], ('Type', 'char', 1, [], 'None')))]
}

func_smth = {
    'start': 1048576,
    'cfg'  : {
        1048576: {'code': [('ASSIGNMENT',
                            ('Variable',
                             'result',
                             ('Type', 'int', 4, [], 'None')),
                            ('Constant', 0, ('Type', 'int', 4, [], 'None'))),
                           ('ASSIGNMENT',
                            ('Variable',
                             'local_c',
                             ('Type', 'int', 4, [], 'None')),
                            ('Constant', 0, ('Type', 'int', 4, [], 'None')))],
                  'ins' : [],
                  'outs': [1048618]},
        1048599: {'code': [('ASSIGNMENT',
                            ('Variable',
                             'result',
                             ('Type', 'int', 4, [], 'None')),
                            ('ADD',
                             ('Variable',
                              'result',
                              ('Type', 'int', 4, [], 'None')),
                             ('Constant', 1, ('Type', 'int', 4, [], 'None')))),
                           ('ASSIGNMENT',
                            ('Variable',
                             'local_c',
                             ('Type', 'int', 4, [], 'None')),
                            ('ADD',
                             ('Variable',
                              'local_c',
                              ('Type', 'int', 4, [], 'None')),
                             ('Constant',
                              1,
                              ('Type', 'int', 4, [], 'None'))))],
                  'ins' : [1048618],
                  'outs': [1048618]},
        1048618: {'code': [('ControlFlowExpression',
                            ('LESS_THAN',
                             ('Variable',
                              'local_c',
                              ('Type', 'int', 4, [], 'None')),
                             ('Constant',
                              10,
                              ('Type', 'int', 4, [], 'None'))))],
                  'ins' : [1048576, 1048599],
                  'outs': [1048626, 1048599]},
        1048626: {'code': [('ASSIGNMENT',
                            ('Variable',
                             'result',
                             ('Type', 'int', 4, [], 'None')),
                            ('Variable',
                             'result',
                             ('Type', 'int', 4, [], 'None'))),
                           ('RETURN',
                            ('Constant',
                             0,
                             ('Type', 'undefined8', 8, [], 'None')))],
                  'ins' : [1048618],
                  'outs': []}},
    'vars' : [('Variable', 'result', ('Type', 'int', 4, [], 'None')),
              ('Variable', 'local_c', ('Type', 'int', 4, [], 'None'))]
}
