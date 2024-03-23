# GRIB2 Reader

Read a GRIB2 file and search for data based on parameter and level values. The results can either be decoded or extracted as a binary blob so it can be saved to a separate file.


# Usage
Add this to your Cargo.toml:

```toml
[dependencies]
grib1_reader = "0.1.0"
```
and this to your source code:

```rust
use grib2_reader::{Grib2Reader, SearchParams};
```
# Example

```rust
let file = File::open("data/sample.grib").await?;
let mut reader = Grib2Reader::new(BufReader::new(file));
let result = reader.read(vec![SearchParams { param: 33, level: 700 }]).await?;

println!("Results:");
for grib in result {
    println!("{:#?}", &grib.pds);
    if let Some(gds) = grib.gds {
        println!("{:#?}", &gds);
    }
}
```