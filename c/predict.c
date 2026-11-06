/* AI-Lab predictor - runs the neural net trained by python/train.py.
 *
 * Reads ../weights.txt (plain text), then evaluates inputs you give it.
 * Zero dependencies - just gcc.
 *
 * Build:  gcc -O2 -o predict predict.c -lm
 * Run:    ./predict            (interactive)
 *         ./predict 0 1        (one-shot)
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_HIDDEN 64
#define MAX_IN 16

static int in_dim, hid_dim;
static double W1[MAX_IN][MAX_HIDDEN], b1[MAX_HIDDEN];
static double W2[MAX_HIDDEN][1],  b2;

static void die(const char *msg) {
    fprintf(stderr, "error: %s\n", msg);
    exit(1);
}

static void load_weights(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) die("cannot open weights.txt - run python/train.py first");

    int r, c;
    /* W1 */
    if (fscanf(f, "%d %d", &r, &c) != 2) die("bad weights file");
    in_dim = r; hid_dim = c;
    if (in_dim > MAX_IN || hid_dim > MAX_HIDDEN) die("network too big");
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            if (fscanf(f, "%lf", &W1[i][j]) != 1) die("bad weights file");

    /* b1 */
    if (fscanf(f, "%d %d", &r, &c) != 2) die("bad weights file");
    if (r != 1 || c != hid_dim) die("bad bias dims");
    for (int j = 0; j < c; j++)
        if (fscanf(f, "%lf", &b1[j]) != 1) die("bad weights file");

    /* W2 */
    if (fscanf(f, "%d %d", &r, &c) != 2) die("bad weights file");
    if (r != hid_dim || c != 1) die("bad W2 dims");
    for (int j = 0; j < hid_dim; j++)
        if (fscanf(f, "%lf", &W2[j][0]) != 1) die("bad weights file");

    /* b2 */
    if (fscanf(f, "%d %d", &r, &c) != 2) die("bad weights file");
    if (r != 1 || c != 1) die("bad b2 dims");
    if (fscanf(f, "%lf", &b2) != 1) die("bad weights file");

    fclose(f);
}

static double predict(const double *x) {
    double hidden[MAX_HIDDEN];
    for (int j = 0; j < hid_dim; j++) {
        double z = b1[j];
        for (int i = 0; i < in_dim; i++)
            z += x[i] * W1[i][j];
        hidden[j] = tanh(z);
    }
    double z2 = b2;
    for (int j = 0; j < hid_dim; j++)
        z2 += hidden[j] * W2[j][0];
    return 1.0 / (1.0 + exp(-z2));
}

int main(int argc, char **argv) {
    /* find weights.txt: ../weights.txt (run from c/) or ./weights.txt */
    const char *path = NULL;
    FILE *probe = fopen("../weights.txt", "r");
    if (probe) { fclose(probe); path = "../weights.txt"; }
    else {
        probe = fopen("weights.txt", "r");
        if (probe) { fclose(probe); path = "weights.txt"; }
    }
    if (!path) die("weights.txt not found - run python/train.py first");

    load_weights(path);
    printf("AI-Lab predictor | net: %d -> %d -> 1\n", in_dim, hid_dim);

    if (argc > 1) {
        /* one-shot: predict 0 1 */
        if (argc - 1 != in_dim)
            die("expected exactly the network's input count");
        double x[MAX_IN];
        for (int i = 0; i < in_dim; i++) x[i] = atof(argv[i + 1]);
        printf("output: %.4f\n", predict(x));
        return 0;
    }

    /* interactive */
    printf("enter %d numbers per line (blank to quit):\n", in_dim);
    char line[512];
    while (fgets(line, sizeof line, stdin)) {
        if (line[0] == '\n' || line[0] == '\r') break;
        double x[MAX_IN] = {0};
        int n = 0;
        char *tok = strtok(line, " \t\n");
        while (tok && n < in_dim) {
            x[n++] = atof(tok);
            tok = strtok(NULL, " \t\n");
        }
        if (n == 0) break;
        if (n != in_dim) {
            printf("need %d inputs, got %d\n", in_dim, n);
            continue;
        }
        printf("output: %.4f\n", predict(x));
    }
    return 0;
}
