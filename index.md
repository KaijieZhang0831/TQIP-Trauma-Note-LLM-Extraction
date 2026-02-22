---
title: Large Language Models for Trauma Care Quality
---

# Introduction

An effective surgical quality program relies heavily on maintaining a comprehensive patient registry to track outcomes and benchmark against national standards. The American College of Surgeons Committee on Trauma requires trauma centers to maintain these registries for accreditation, which are critical for reporting trauma quality metrics. However, traditional registries are labor-intensive and costly. Artificial intelligence, particularly Large Language Models (LLMs), offers a potential solution to streamline this process. We hypothesized that a LLM could be applied to review patient charts and identify complications as defined by the Trauma Quality Improvement Program, offering an effective adjunct to manual chart reviews.

# Methods

Our primary outcome was agreement between the LLM and manual reviews performed by the institution's trauma registry. We assessed sensitivity, negative predictive value (NPV), positive predictive value (PPV), and frequency of complications identified by the LLM but not the registrar. Additionally, a subset of cases and output from the LLM was reviewed and validated by clinical subject matter experts (SMEs) via manual chart review. Cases in which the LLM missed complications that the human registrars identified were specifically included in this subset. Rationale provided by the LLM facilitated targeted reviews and verification of output as true or false.

![Diagram of LLM Retrieval](assets/Intro_Pipe.jpg)

# Results

## Performance

As a simple smoke test for AWS Bedrock with new embedding and LLMs, here is the pre-Embedding and pre-LLM optimization (raw) of experiment on 20 patient features (During extended runs we encountered several technical issues that required additional time to diagnose and stabilize, therefore we used a subset of 20 samples as a preliminary peek at the system behavior): TODO: Insert Table

# Conclusion

Overall, our progress is still slightly behind the original timeline, but we have completed the most difficult portion of the work. Next, our direction is clear. We will first optimize the current LLM setup and align its parameters with our prior LLaMA-based experiments to achieve comparable performance. We will then run ablation studies by swapping in the new components we proposed, so we can isolate their effects and draw concrete conclusions.

# The Team

- Viv Somani
- Kaijie Zhang
- Aaron Boussina (mentor)
