# Livecodebench Errata
While we have tries to ensure the correctness of the benchmark in terms of test cases and problems, we have received feedback on issues regarding erroneous tests and problems not amenable to autograding. Here, we document the known issues and are also constantly using this feedback to improve our problem selection heuristics as we update LiveCodeBench.

## Multiple Solutions Accepted
7 problems have been identified with test case issues. Particularly, these problems accept multiple possible outputs, while the benchmark grades for only one specific output. Thus some correct solutions may be marked as incorrect and can add noise to the benchmark results.

1. abc311_c - Multiple solutions accepted
2. abc326_d - Multiple solutions accepted
3. abc327_b - Multiple solutions accepted
4. abc333_e - Multiple solutions accepted
5. abc343_e - Multiple solutions accepted
6. abc362_c - Multiple solutions accepted
7. find-words-containing-character - Multiple solutions accepted
8. find-the-peaks - Multiple solutions accepted
10. generate-binary-strings-without-adjacent-zeros - Multiple solutions accepted


## Interactive Problems
2 problems have been identified as interactive problems. These problems require the submission to interact with the judge to get the final answer. The benchmark evaluation suite does not support interactive problems and thus these problems cannot be solved correctly. Note that these problems will not affect model comparison results since no model can solve these problems.

1. abc337_e - Interactive problem
2. abc355_e - Interactive problem

## Erroneous Test Cases
1 problem has been identified with erroneous test cases during scraping. This problem cannot be solved correctly with the current test cases. Note that these problems will not affect model comparison results since no model can solve these problems.

1. abc350_c - Erroneous test cases