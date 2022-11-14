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
