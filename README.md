# GRIB2 Reader
This is designed to do two things:

 1. Cut up a combined grib2 file into smaller individual grib2 parts using tokio and async.
 2. Parse a single grib2 file from a `Vec<u8>` blob (without tokio and async).

# Usage
Add this to your Cargo.toml if you want to use the async feature:

```toml
[dependencies]
grib2_reader = { version = "0.1.2", features = ["async"] }
```

Add this to your Cargo.toml if you only want to parse a single grib2 file:

```toml
[dependencies]
grib2_reader = "0.1.2"
```

# Example
Count the number of individual grib2 files inside a combined grib2 file.
```rust
use grib2_reader::{Grib2Reader, Grib2Error};

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
When you just want to parse a single grib2 file:
```rust
use grib2_reader::{Grib2Parser, Grib2Error};

fn parse_single() {
    let mut f = File::open("data/HARMONIE_DINI_SF_5.grib").expect("Unable to open file");

    let mut data = vec![];
    f.read_to_end(&mut data).expect("Unable to read file");

    let mut grib2_parser = Grib2Parser::new();
    let mut grib = grib2_parser.parse(data).expect("Unable to parse grib2 file");

    println!("Results:");
    // We don't want to display the binary data, so remove that from the output
    grib.data[0].data = vec![];
    println!("{:#?}", &grib);
}
```