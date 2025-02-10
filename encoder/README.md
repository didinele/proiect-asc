# encoder

Utilizare:

```sh
$ yarn build
```

Chiar daca sunt build errors, pot fi ignorate, exista un output in `dist/index.js`.

```sh
$ node dist/index.js
```

Detalii tehnice:
- folosim hardcoded `byte` mode (potrivit pentru tipurile de input cu care vom lucra)
- folosim hardcoded `h` error correction
