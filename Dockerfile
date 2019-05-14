FROM rust:1.34

WORKDIR /usr/src

# Host dependencies
RUN apt-get update && apt-get install -y libclang-4.0-dev && apt-get clean

# Build dependencies
COPY Cargo.toml ./
RUN mkdir src && touch src/lib.rs && cargo build

# Run tests
COPY . .
RUN cargo test -- --nocapture --test-threads=1
