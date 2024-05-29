<a href="https://colab.research.google.com/github/dcownden/PerennialProblemsOfLifeWithABrain/blob/main/sequences/P1C3_RealEvolution/student/P1C3_Title.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> &nbsp; <a href="https://kaggle.com/kernels/welcome?src=https://raw.githubusercontent.com/dcownden/PerennialProblemsOfLifeWithABrain/main/sequences/P1C3_RealEvolution/student/P1C3_Title.ipynb" target="_parent"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open in Kaggle"/></a>

The following is part of a test for an upcoming text book on computational neuroscience from an optimization and learning perspective. The book will start with evolution because ultimately, all aspects of the brain are shaped by evolution and, as we will see, evolution can also be seen as an optimization algorithm. We are sharing it now to get feedback on what works and what does not and the developments we should do.

___
# Chapter 1.3 Real Evolution

### Objective:
In the previous chapters we learned to describe and think about behaviour as a policy, to evaluate an organism's behaviour relative to the goals implied by its environmental niche, and to use a variety of optimization algorithms to improve an organisms behaviour relative to its niche, i.e. to adapt the behaviour to the environment. In this next chapter we are going to look at how evolutionary processes, i.e. natural selection for survivability and reproduction, provide a powerful framework for understanding much of the observable universe, neuroscience included!

You will learn:
*   What an evolutionary process is and how it provides a structure for answering fundamental why questions. Primarily question about why we observe the world in this one state and not another.
*   What evolutionary processes are - selection applied to variation in a population - and how they are a form of optimization.
*   Why sex is so important.
*   How competition and interaction complicate optimization. Optimization in a strict sense does not directly apply to situations with multiple interacting organisms, each with their own goals that they are simultaneously optimizing for. Ideas from Game Theory and dynamical systems are required to do multi-organism 'optimization' the right way.
*   How learning and evolution interact as parts of a nested optimization loop (Learning -> Inner, Evolution -> Outer), that can produce adaptive behaviour more efficiently than either process in isolation.     

### Context
This chapter is the third of four in the first part of the book. The first part of the book is about **Behaviour, Environments and Optimization: Evolution and Learning**

***Animals are adapted to their specific environments; their behaviour is best understood within the context of their evolutionary environment.***


Part 1 of the book aims to introduce the fundamental concepts of
* **Environment**, where an organism lives
* **Behaviour**, what the organism does in the environment
* **Optimization**, how learning and evolution shape an organism's behaviour to make it better suited to its environment

This is the core of why we are writing this book: pretty much anything happening in the brain (and biology) can be viewed as part of a process that brings about improvement in this sense. In this first part of the book we set out this foundational perspective. Each subsequent part shows how this perspective connects insights from Machine Learning to the function of the brain in a way that can both synthesize and guide empirical neuroscience research.

