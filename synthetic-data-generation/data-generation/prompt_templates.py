comedy_template = '''{persona}
 
Assume you are the persona described above and I want you to act as a stand-up comedian. Write content that reflects your unique voice, expertise, and humor, tailored to your specific field. 
'''

instruction_template = '''Guess a prompt that the following persona may ask you to do:

{persona}

Note:

1. The prompt should be informative and specific.
2. Your output should start with "User prompt:"'''

math_template = '''Create a math problem related to the following persona:

{persona}

Note:

1. The math problem should be challenging and involve advanced mathematical skills and knowledge. Only top talents can solve it correctly.
2. You should make full use of the persona description to create the math problem to ensure that the math problem is unique and specific to the persona.
3. Your response should always start with "Math problem:". Your response should not include a solution to the created math problem.
4. Your created math problem should include no more than 2 sub-problems.
'''

knowledge_template = '''{persona}

Assume you are the persona described above and you are writing a Quora article using your knowledge, skills, experience, or insights to help others learn and benefit from it.

Note:

1. The article should be specific, informative and knowledge-rich.
2. Your response should start with "Title:"'''