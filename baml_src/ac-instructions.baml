class Instructions {
    steps string[]
    page int
    warnings string[]
}

function ProcessInstructions {
    input (query: string, answer: string)
    output Instructions
}

impl<llm, ProcessInstructions> V1 {
    client GPT4
    prompt #"
        {// {#chat(system)}
        Answer the question:
        {#input.query} //}

        {#input.answer}

        Convert to this Output JSON Format:
        {#print_type(output)}
    "#
}