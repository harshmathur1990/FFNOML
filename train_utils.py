import torch


def extract_temperature(X):
    """
    X: (B, C, Nz, k, k)
       channel 0 is assumed to be log10(T)
    returns:
       T: (B, Nz)
    """
    k = X.shape[-1]
    center = k // 2

    logT = X[:, 0, :, center, center]   # (B, Nz)
    T = 10.0 ** logT                    # convert log10(T) -> T

    return T


def compute_loss(pred, target, weight, loss_fn, T):

    loss, components = loss_fn(T, pred, target)   # (B)

    if weight is not None:
        loss = loss * weight

    loss = loss.mean()

    return loss, components


def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    scaler,
    device,
    amp=True,
    grad_clip=1.0,
    log_every=50,
):

    model.train()
    running = 0.0

    for it, (x, y, dx, dy, scale, weight) in enumerate(loader, start=1):

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        dx = dx.to(device, non_blocking=True)
        dy = dy.to(device, non_blocking=True)

        if weight is not None:
            weight = weight.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(amp and device.startswith("cuda"))):

            pred = model(x, dx, dy)

            # ---- extract T from input cube ----
            T = extract_temperature(x)

            loss, components = compute_loss(
                pred=pred,
                target=y,
                weight=weight,
                loss_fn=loss_fn,
                T=T,
            )

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if grad_clip is not None and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        running += loss.item()

        if (it % log_every) == 0:
            avg = running / it

            if components is not None and isinstance(components, dict):
                msg = f"iter {it:05d}/{len(loader)} train_loss={avg:.6e}"

                if "data" in components:
                    msg += f" data={components['data'].item():.6e}"
                if "source" in components:
                    msg += f" source={components['source'].item():.6e}"

                if "source_per_atom" in components and "atom_names" in components:
                    spa = components["source_per_atom"]
                    names = components["atom_names"]
                    atom_terms = ", ".join(
                        f"{n}={spa[j].item():.3e}" for j, n in enumerate(names)
                    )
                    msg += f" [{atom_terms}]"

                print(msg)
            else:
                print(f"iter {it:05d}/{len(loader)} train_loss={avg:.6e}")

    return running / max(1, len(loader))


def validate(
    model,
    loader,
    loss_fn,
    device,
    amp=True,
):

    model.eval()

    tot = 0.0
    n = 0

    with torch.no_grad():

        for x, y, dx, dy, scale, weight in loader:

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            dx = dx.to(device, non_blocking=True)
            dy = dy.to(device, non_blocking=True)

            if weight is not None:
                weight = weight.to(device, non_blocking=True)

            # ---- extract T from input cube ----
            T = extract_temperature(x)

            with torch.cuda.amp.autocast(enabled=(amp and device.startswith("cuda"))):

                pred = model(x, dx, dy)

                loss, _ = compute_loss(
                    pred=pred,
                    target=y,
                    weight=weight,
                    loss_fn=loss_fn,
                    T=T,
                )

            tot += loss.item()
            n += 1

    return tot / max(1, n)


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    scaler,
    save_path,
    *,
    num_epochs=50,
    device="cuda",
    amp=True,
    grad_clip=1.0,
    early_stopping=None,
):

    best_val = float("inf")

    if early_stopping is not None:
        es_enabled = early_stopping.get("enabled", False)
        patience = early_stopping.get("patience", 10)
        min_delta = early_stopping.get("min_delta", 0.0)
    else:
        es_enabled = False
        patience = 0
        min_delta = 0.0

    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            scaler=scaler,
            device=device,
            amp=amp,
            grad_clip=grad_clip,
        )

        if val_loader is not None:

            val_loss = validate(
                model=model,
                loader=val_loader,
                loss_fn=loss_fn,
                device=device,
                amp=amp,
            )

            improved = (best_val - val_loss) > min_delta

            if improved:

                best_val = val_loss
                epochs_no_improve = 0

                torch.save(
                    dict(
                        epoch=epoch,
                        model_state=model.state_dict(),
                        opt_state=optimizer.state_dict(),
                        train_loss=train_loss,
                        val_loss=val_loss,
                    ),
                    save_path,
                )

                print(
                    f"[Epoch {epoch:03d}] "
                    f"train={train_loss:.6e} "
                    f"val={val_loss:.6e} (saved best)"
                )

            else:

                epochs_no_improve += 1

                print(
                    f"[Epoch {epoch:03d}] "
                    f"train={train_loss:.6e} "
                    f"val={val_loss:.6e}"
                )

                if es_enabled and epochs_no_improve > patience:
                    print(f"Early stopping triggered (patience={patience})")
                    break

        else:

            torch.save(
                dict(
                    epoch=epoch,
                    model_state=model.state_dict(),
                    opt_state=optimizer.state_dict(),
                    train_loss=train_loss,
                ),
                save_path,
            )

            print(f"[Epoch {epoch:03d}] train={train_loss:.6e} (saved)")