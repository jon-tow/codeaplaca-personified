import json
import datasets


def build_code_personas(
    dataset_name: str = "argilla/FinePersonas-v0.1",
    push_to_hub: bool = False
) -> datasets.Dataset:
    dataset = datasets.load_dataset(dataset_name, split="train")
    # Remove `embeddings` column for storage efficiency
    dataset = dataset.remove_columns(["model_name_embeddings", "embedding"])
    # Only keep the programming-related labels...
    # Ref: https://huggingface.co/datasets/argilla/FinePersonas-v0.1#dataset-summary
    labels = [
        ["Computer Networking", "Cybersecurity", "Technology"],
        ["Technology", "Research", "Artificial Intelligence"],
        ["Engineering", "Technology", "Computer Science"],
        ["Education", "Computer Science", "Teacher/Instructor"],
        ["Tech Professional", "Web Developer", "IT Specialist"],
        ["Education", "Computing", "Teaching"],
        ["Data Analysis", "Statistical Expertise", "R Programming"],
        ["Computer Science", "Graphics", "Technology"],
        ["Database Professional", "IT Specialist", "Software Developer"],
        ["Educator", "Programmer", "Technologist"],
    ]
    labels = [json.dumps(label) for label in labels]
    persona_dataset = dataset.filter(
        lambda x: any(x["labels"] == label for label in labels)
    )
    if push_to_hub:
        persona_dataset.push_to_hub("jon-tow/FineProgrammingPersonas-v0.1")
    return persona_dataset


if __name__ == "__main__":
    build_code_personas()
