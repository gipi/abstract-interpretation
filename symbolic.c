#include <stdlib.h>


extern int result;

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


void smth() {
    result = 0;
    for (int i = 0 ; i < 10 ; i++)
        result += 1;
}

int goto_the_moon(int count) {
    int result = 0;

    result++;

    goto miao;

    result++;

        miao:
    for (int i = 0 ; i < count ; i++) {
        result--;
        result /= 2;
    }

    return result;
}

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


#define SIZE_PROMPT 5
#define COUNT_MIAOS 5

struct miao {
    int index;
    char prompt[SIZE_PROMPT + 1];
};

struct miao* init_miao() {
    struct miao* miaos = (struct miao*)malloc(COUNT_MIAOS *  sizeof(struct miao));

    for (int cycle = 0 ; cycle < COUNT_MIAOS ; cycle++) {
        struct miao* bau = &miaos[cycle];

        bau->index = cycle;

        for (int count = 0 ; count < SIZE_PROMPT ; count++) {
            bau->prompt[count] = '0' + cycle;
        }

        bau->prompt[SIZE_PROMPT] = '\0';
    }

    return miaos;
}

int do_something_w_struct(struct miao*, int);

void call_me_w_a_struct() {
    struct miao bau = {
        .index = 0,
        .prompt = "kebab"
    };

    do_something_w_struct(&bau, 3);
}

char* get_prompt(struct miao m) {
    return m.prompt;
}

char* get_ptr_prompt(struct miao* m) {
    return m->prompt;
}

char get_ptr_prompt_at(struct miao* m, unsigned int index) {
    return m->prompt[index];
}

int get_ptr_index(struct miao* m) {
    return m->index;
}
