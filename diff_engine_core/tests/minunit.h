#ifndef MINUNIT_H
#define MINUNIT_H
/* Modified from http://www.jera.com/techinfo/jtns/jtn002.html */

/* Simple Macros for testing */
#define mu_assert(message, test)                                                    \
    do                                                                              \
    {                                                                               \
        if (!(test)) return message;                                                \
    } while (0)

#define mu_assert_less(message, a, b)                                               \
    do                                                                              \
    {                                                                               \
        if (a > b)                                                                  \
        {                                                                           \
            printf("%s: %1.3e > %1.3e\n", message, a, b);                           \
            return message;                                                         \
        }                                                                           \
    } while (0)

#define mu_run_test(test, var) _mu_run_test(#test, test, var)

#define _mu_run_test(name, test, var)                                               \
    do                                                                              \
    {                                                                               \
        printf("*********************************************************\n");      \
        printf("Running test: %s\n", name);                                         \
        printf("*********************************************************\n");      \
        const char *message = test();                                               \
        var++;                                                                      \
        if (message)                                                                \
        {                                                                           \
            printf("FAILED: %s\n", message);                                        \
            return 1;                                                               \
        }                                                                           \
        printf("PASSED\n\n");                                                       \
    } while (0)

#endif
