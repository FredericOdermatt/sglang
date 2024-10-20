"""
Usage:
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000
python readme_examples.py
"""

import matplotlib.pyplot as plt
import numpy as np
import sglang as sgl


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


@sgl.function
def text_qa(s, question):
    s += "Q: " + question + "\n"
    s += "A:" + sgl.gen("answer", choices=["yes", "no"])


@sgl.function
def text_qa_with_long_systemprompt(s, question):
    s += """
Your job is to score the following questions based on whether they should be answered with "yes" or not. Output "yes" if the answer is clearly "yes" based on common knowledge, and "no" if it should be answered with "no."

The following question is correct: Is water wet?
Here is the same question, but not correct: Is water dry?

The following question is correct: Is the sky blue on a clear day?
Here is the same question, but not correct: Is the sky green?

The following question is correct: Is the Earth round?
Here is the same question, but not correct: Is the Earth flat?

The following question is correct: Is the sun a star?
Here is the same question, but not correct: Is the sun a planet?

The following question is correct: Is fire hot?
Here is the same question, but not correct: Is fire cold?

The following question is correct: Is banana a fruit?
Here is the same question, but not correct: Is banana a vegetable?

The following question is correct: Is Mount Everest the tallest mountain in the world?
Here is the same question, but not correct: Is Mount Everest underwater?

Give an output for the following question:  
"""
    s += question
    s += "\n"
    s += sgl.gen("answer", choices=["yes", "no"])


def driver_batching():
    avg_yes_logits = []
    avg_no_logits = []
    for _ in range(100):
        states = text_qa_with_long_systemprompt.run_batch(
            [
                {"question": "Is summer typically hot?"},  # 40 Yes questions
                {"question": "Is grass green?"},
                {"question": "Is a bird an animal?"},
                {"question": "Is salt a mineral?"},
                {"question": "Is iron a metal?"},
                {"question": "Is rain wet?"},
                {"question": "Is a bicycle a form of transportation?"},
                {"question": "Is New York a city in the United States?"},
                {"question": "Is air necessary for life?"},
                {"question": "Is hydrogen the most abundant element in the universe?"},
                {"question": "Is the speed of light constant in a vacuum?"},
                {"question": "Is DNA the molecule that carries genetic information in humans?"},
                {"question": "Is the Pythagorean theorem applicable in right-angled triangles?"},
                {"question": "Is the Eiffel Tower located in Paris?"},
                {"question": "Is the boiling point of water 100°C at sea level?"},
                {"question": "Is the mitochondrion the powerhouse of the cell?"},
                {"question": "Is gravity responsible for keeping planets in orbit around the sun?"},
                {
                    "question": "Is the number of planets in our solar system currently considered to be eight?"
                },
                {"question": "Is a prime number only divisible by 1 and itself?"},
                {"question": "Is quantum mechanics a fundamental theory in physics?"},
                {"question": "Is the human genome composed of DNA?"},
                {"question": "Is the speed of sound faster in water than in air?"},
                {"question": "Is Antarctica the largest desert on Earth?"},
                {"question": "Is carbon the basis of organic life?"},
                {"question": "Is the Moon tidally locked to the Earth?"},
                {"question": "Is Schrödinger's cat an example of quantum superposition?"},
                {"question": "Is the universe still expanding?"},
                {"question": "Is a black hole's event horizon a point of no return?"},
                {
                    "question": "Is the kilogram a unit of mass in the International System of Units?"
                },
                {"question": "Is it possible for a human to perceive sound in a vacuum?"},
                {"question": "Is the theory of relativity associated with Albert Einstein?"},
                {"question": "Is the chemical formula for water H2O?"},
                {"question": "Is the square root of 4 equal to 2?"},
                {"question": "Is Pluto classified as a dwarf planet?"},
                {"question": "Is string theory still a speculative framework in physics?"},
                {"question": "Is photosynthesis necessary for most plants to produce energy?"},
                {"question": "Is light both a particle and a wave?"},
                {"question": "Is the second law of thermodynamics related to entropy?"},
                {"question": "Is there an infinite number of prime numbers?"},
                {
                    "question": "Is time travel to the past considered physically impossible by general relativity?"
                },
                {"question": "Is the Moon made of cheese?"},  # 40 NO QUESTIONS
                {"question": "Is the Sun colder than the Earth?"},
                {"question": "Is sound able to travel in a vacuum?"},
                {
                    "question": "Is the Great Wall of China visible from the Moon with the naked eye?"
                },
                {"question": "Is Mars closer to the Sun than Venus?"},
                {"question": "Is 2 + 2 equal to 5?"},
                {"question": "Is the speed of light slower than the speed of sound?"},
                {"question": "Is water flammable?"},
                {"question": "Is the square root of 9 equal to 5?"},
                {"question": "Is Antarctica home to a permanent human population?"},
                {"question": "Is a kilometer longer than a mile?"},
                {"question": "Is zero a positive number?"},
                {"question": "Is Mount Everest located in North America?"},
                {"question": "Is the Sahara Desert located in South America?"},
                {"question": "Is gold a gas at room temperature?"},
                {"question": "Is a triangle with sides 3, 4, and 8 possible?"},
                {"question": "Is a human's normal body temperature below 30°C?"},
                {"question": "Is lightning slower than a human running?"},
                {"question": "Is the human body made primarily of metal?"},
                {"question": "Is a square a type of triangle?"},
                {"question": "Is the mass of an electron greater than that of a proton?"},
                {"question": "Is absolute zero a temperature that can be reached in practice?"},
                {"question": "Is the concept of infinity a number in conventional mathematics?"},
                {"question": "Is time dilation a phenomenon that only occurs at low speeds?"},
                {"question": "Is the Earth at the center of the Milky Way galaxy?"},
                {"question": "Is every even number also a prime number?"},
                {"question": "Is dark matter directly observable with current technology?"},
                {"question": "Is an isotope a particle with no mass?"},
                {
                    "question": "Is a vacuum completely devoid of all particles, including virtual particles?"
                },
                {"question": "Is the Heisenberg uncertainty principle irrelevant at large scales?"},
                {"question": "Is the charge of a neutron positive?"},
                {"question": "Is entropy a measure of energy creation in a closed system?"},
                {"question": "Is the second derivative of a constant function non-zero?"},
                {
                    "question": "Is the Big Bang theory an explanation for the creation of the universe from nothing?"
                },
                {"question": "Is the force of gravity on Earth constant across all locations?"},
                {
                    "question": "Is it possible to create perpetual motion machines according to the laws of thermodynamics?"
                },
                {"question": "Is the speed of light variable in a vacuum?"},
                {"question": "Is uranium the only element used in nuclear fission reactors?"},
                {"question": "Is the human brain composed primarily of solid bone?"},
                {"question": "Is teleportation of matter a proven and widely used technology?"},
                {
                    "question": "Is it reasonable to expect a human to survive in the vacuum of space without a spacesuit?"
                },
            ],
            progress_bar=True,
            temperature=0.0,
        )

        # print(states[0])
        yes_logits = []
        no_logits = []
        for idx, state in enumerate(states):
            meta_info = state.get_meta_info("answer")
            choice_1_logit = meta_info["input_token_logprobs"][0][0][0]
            choice_2_logit = meta_info["input_token_logprobs"][1][0][0]
            choice_1_logprobs, _ = softmax([choice_1_logit, choice_2_logit])
            if idx <= 40:
                yes_logits.append(choice_1_logprobs)
            else:
                no_logits.append(choice_1_logprobs)
        print(sum(yes_logits) / len(yes_logits))
        print(sum(no_logits) / len(no_logits))
        avg_yes_logits.append(sum(yes_logits) / len(yes_logits))
        avg_no_logits.append(sum(no_logits) / len(no_logits))

    plt.figure()
    plt.plot(avg_yes_logits, label="yes")
    plt.plot(avg_no_logits, label="no")
    plt.legend()
    plt.ylim(0, 1)
    plt.title("40 yes followed by 40 no questions")
    plt.savefig("yes_no_logits.png")


if __name__ == "__main__":
    # sgl.set_default_backend(sgl.OpenAI("gpt-3.5-turbo-instruct"))
    # sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30001"))
    sgl.set_default_backend(sgl.RuntimeEndpoint("http://127.0.0.1:30001"))
    driver_batching()
