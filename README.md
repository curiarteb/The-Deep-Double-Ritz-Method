# The-Deep-Double-Ritz-Method

In this submission, I present three self-developed deep-learning-based implementations in TF2 of the methods: WANs (Weak Adversarial Networks), GDRM (The Generalized Deep Ritz Method), and D2RM (The Deep Double Ritz Method). These methods are discussed in the article titled "A Deep Double Ritz Method for solving Partial Differential Equations using Neural Networks". You can find an open-source version of the article here: https://www.sciencedirect.com/science/article/pii/S0045782523000154

For clarity and modularity, I have encoded each method in separate and independent modules: WANs, GDRM, and D2RM. The results of the polynomial cases from the article can be easily reproduced using this initial version. However, I must note that for the other cases, additional complexities exist, as explained in the article. These complexities involve incorporating very specific subtleties that are case-specific (like the midpoint integration rules with a stochastic integration mesh, specifically designed to capture the singularities present in each case). Regrettably, these case-by-case intricacies are not yet included in this first version.

As a non-developer, I recognize that my first coding to produce the results of the published article was anything but user-friendly. With the intention to produce a user-friendly version, I believe this first base launch represents a somewhat elegant solution for now. Your understanding and patience are greatly appreciated.

For any specific inquiries or interests regarding this work or related topics, please feel free to contact me directly at carlos.uribar@gmail.com. I am open to discussions and happy to provide further insights. Thank you for your understanding and interest in my research.





