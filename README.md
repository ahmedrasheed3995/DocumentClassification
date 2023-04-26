# Document classifier

## Run with Docker
Start with building the docker image. Go in to the project directory, open terminal in that directory and use the following command 

```bash
docker image build -t layoutmv3 .
```

Then Run the docker image

```bash
docker run --rm -v <local-image-dir-path>:/image_dir/ --env IMAGE='<image-name>' layoutmv3:latest inference.py
```

For example
```bash
docker run --rm -v /Users/ahmedrasheed/PycharmProjects/LayoutLMv3/resume:/image_dir/ --env IMAGE='doc_000051.png' layoutmv3:latest inference.py
```
output
```bash
{'resume': 0.9814807772636414, 'scientific_publication': 0.007665990386158228, 'email': 0.010853313840925694}
```
