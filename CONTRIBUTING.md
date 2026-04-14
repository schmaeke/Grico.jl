# Contributing to Grico.jl

This repository is a research-software code base. Contributions should favor
clarity, mathematical correctness, compact implementation, and durable design
over short-term convenience. In particular, public APIs and nontrivial internal
algorithms should be documented well enough that a new reader can recover both
the numerical intent and the implementation strategy from the source.

## General Workflow

- Keep changes scoped. Avoid bundling unrelated cleanups into a feature or bug
  fix unless they are necessary for a coherent design.
- Preserve the architectural direction of the rewrite. If the existing structure
  stands in the way of a clean solution, improve the structure rather than
  layering on a workaround.
- Run formatting and relevant tests before concluding a change.

```bash
julia format.jl
julia --project=. -e 'using Pkg; Pkg.test()'
```

The package root intentionally tracks only [`Project.toml`](Project.toml), not
`Manifest.toml`. The [`benchmark/`](benchmark) environment keeps its own
[`Project.toml`](benchmark/Project.toml), while local manifests and generated
benchmark reports stay untracked.

- Add or update tests when behavior changes.
- If a change affects exported API, user-facing examples, or mathematical
  behavior, update the corresponding documentation and comments in the same
  change.
- Do not leave stale comments behind. If implementation and prose disagree, the
  prose is wrong until it is updated.

## Documentation and Commenting Style

The library should be readable at two levels at once:

- A student should be able to understand what mathematical object, operator, or
  algorithm a piece of code represents.
- An advanced reader should be able to recover the main numerical choices,
  invariants, and implementation strategy without reverse-engineering the code
  line by line.

Comments and docstrings are therefore part of the implementation, not optional
afterthoughts.

## General Principles

- Write in full sentences and connected prose. Prefer explanatory paragraphs to
  label-like fragments.
- Explain what the code represents, why it is written this way, and what
  assumptions it relies on.
- Keep documentation close to the code it describes.
- Prefer precise scientific language over conversational or promotional prose.
- Define symbols and terminology before using them freely.
- Avoid duplicating the same explanation in multiple places. Put the main
  contract at the public entry point and use internal comments for local
  implementation details.
- Do not comment obvious syntax. Comments should explain meaning, invariants,
  numerical structure, data layout, or non-obvious control flow.
- Small and genuinely self-evident helpers may remain undocumented internally if
  a comment would add no information.

## Exported API: Julia Docstrings

Every exported symbol should have a classic Julia docstring placed immediately
above the documented definition. This includes exported types, constructors,
functions, and user-facing constants when their meaning is not already obvious.

Docstrings should usually follow this pattern:

1. A short opening sentence stating what the object is or what the function
   does.
2. One or more paragraphs describing the mathematical or algorithmic meaning.
3. A paragraph describing how the object should be used, including important
   assumptions, invariants, or limitations.
4. Optional details on arguments, return values, dimensions, indexing
   conventions, or performance characteristics when those details are relevant.

For exported types:

- Explain what mathematical or structural object the type represents.
- State important invariants or consistency requirements.
- If construction performs validation, mention the validated conditions.

For exported functions:

- Explain the operation in mathematical or algorithmic terms, not only in terms
  of input and output plumbing.
- State the expected interpretation of arguments such as cells, leaves, axes,
  quadrature shapes, or state vectors.
- Mention whether the function allocates, mutates, or assumes prevalidated
  inputs if that matters in practice.

For overloaded generic functions with multiple methods:

- Prefer documenting the generic operation once, then add method-specific notes
  only where behavior materially differs.

## Internal Code: Leading Comments

Internal functions, helper types, and blocks of build logic should receive
ordinary line comments when they are not immediately self-explanatory. These
comments should be placed directly above the corresponding definition or code
block.

Leading comments for internal code should usually answer the following:

- What role does this routine play in the larger algorithm?
- What data or invariants does it rely on?
- What transformation, elimination, restriction, or assembly step is being
  carried out?
- Why is this approach used instead of a simpler but incorrect or less robust
  alternative?

For internal data structures, describe:

- what the stored arrays or fields mean,
- what indexing conventions they use,
- and which invariants other code may assume about them.

## In-Function Comments

Use comments inside a function when the implementation contains a non-obvious
phase, formula, invariant, or indexing transformation.

Good in-function comments explain things such as:

- why a loop is organized in a particular order,
- how a geometric or algebraic mapping is constructed,
- why a constraint elimination step is valid,
- where a numerical stabilization term comes from,
- or which quantity a precomputed cache actually represents.

Avoid comments that merely narrate assignments or restate the code in prose.
The goal is to reduce cognitive load, not to duplicate syntax.

## Mathematical Style

Prefer unicode mathematical notation in source comments and docstrings when it
improves readability. Typical examples are `Ω`, `ξ`, `Δt`, `∇`, `L²`, `C⁰`, and
expressions such as `u · ∇v` or `∫Ω f dΩ`.

In practice:

- Prefer short formulas embedded in prose over large display-style derivations.
- Prefer source-readable unicode notation over LaTeX-style markup in comments
  and docstrings.
- Use notation consistently across files. The same symbol should not silently
  change meaning from one module to another.
- Introduce symbols before using them. A reader should not have to infer what
  `ξ`, `τ`, or `ν` means from distant context.
- In comments and docstrings, mathematical symbols may follow the notation of
  the underlying equations. In identifiers, prefer descriptive ASCII names by
  default and reserve unicode variable names for small, local quantities where
  the mathematical meaning is immediate.

If generated documentation later requires different formatting conventions, that
can be handled separately. Source readability remains the default priority.

## Audience and Tone

Write in a scientific style appropriate for research software:

- concise, precise, and technically explicit,
- careful about assumptions and terminology,
- and readable without being terse to the point of obscurity.

A useful pattern is to begin with a plain-language explanation and then refine
it with the mathematical or algorithmic details. This keeps the code accessible
to students without flattening the content for advanced readers.

## What To Avoid

- Line-by-line narration of obvious code.
- Vague comments such as "helper function" or "do assembly here".
- Restating a function name instead of explaining its contract.
- Copying mathematical text into several places and letting the copies drift.
- Introducing notation in comments that does not match the implementation.
- Long comments that never identify the actual invariant or algorithmic step.

## Review Checklist

Before concluding a change, check the following:

- Every exported symbol touched by the change has an adequate Julia docstring.
- Every nontrivial internal routine touched by the change has a useful leading
  comment if the implementation is not self-evident.
- Complex blocks inside functions are commented at the point where the reader
  needs the explanation.
- Mathematical notation is consistent and defined on first use.
- Comments describe the implemented algorithm, not an earlier version.
- Documentation and comments remain compact enough to support reading rather
  than obstruct it.
