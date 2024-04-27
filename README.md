# GRIB2 Reader
This is designed to do two things:

 1. Cut up a combined grib2 file into smaller individual grib2 parts using tokio and async.
 2. Parse a single grib2 file from a `Vec<u8>` blob (without tokio and async).

# Usage
Add this to your Cargo.toml if you want to use the async feature:

```toml
[dependencies]
grib2_reader = { version = "0.1.0", features = ["async"] }
```

Add this to your Cargo.toml if you only want to parse a single grib2 file:

```toml
[dependencies]
grib2_reader = "0.1.0"
```

and this to your source code:

```rust
use grib2_reader::{Grib2Reader, Grib2Error};
```
# Example
Count the number of individual grib2 files inside a combined grib2 file.
```rust
async fn read_all() -> Result<(), Grib2Error> {
        let f = File::open("data/HARMONIE_DINI_SF_5.grib").await?;

        let mut b_reader = BufReader::new(f);
        let file_length = b_reader.seek(SeekFrom::End(0)).await?;
        let mut reader = Grib2Reader::new(b_reader);

        let mut count = 0;
        loop {
            match reader.read_binary_next(file_length).await {
                Ok(data) => {
                    if data.is_empty() {
                        println!("All done");
                        break;
                    }

                    // Here data is a Vec<u8> of the contained grib2 data

                    count += 1;
                }
                Err(Grib2Error::EOF) => {
                    println!("EOF");
                    break;
                }
                Err(err) => {
                    println!("Err: {:?}", err);
                    break;
                }
            };
        }

        println!("File count: {}", count);

        Ok(())
    }
```